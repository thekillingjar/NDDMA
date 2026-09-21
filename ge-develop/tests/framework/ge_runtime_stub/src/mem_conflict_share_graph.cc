/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/mem_conflict_share_graph.h"
#include <limits>
#include "runtime/mem.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "common/ge_inner_attrs.h"
#include "framework/common/types.h"
#include "graph/utils/graph_utils_ex.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "common/ge_common/ge_types.h"

namespace ge {
namespace {
void AddParentIndexForSubGraphNetoutput(ComputeGraphPtr &root_graph) {
  for (auto &node : root_graph->GetAllNodes()) {
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
}

void SetRtsSpecialTypeInput(ComputeGraphPtr &root_graph, const std::string &node_name, const int64_t type) {
  for (auto &node : root_graph->GetAllNodes()) {
    if (node->GetName() == node_name) {
      std::vector<int64_t> mem_type(node->GetOpDescBarePtr()->GetInputsSize(), type);
      (void)ge::AttrUtils::SetListInt(node->GetOpDescBarePtr(), ATTR_NAME_INPUT_MEM_TYPE_LIST, mem_type);
    }
  }
}

void SetRtsSpecialTypeOutput(ComputeGraphPtr &root_graph, const std::string &node_name, const int64_t type) {
  for (auto &node : root_graph->GetAllNodes()) {
    if (node->GetName() == node_name) {
      std::vector<int64_t> mem_type(node->GetOpDescBarePtr()->GetOutputsSize(), type);
      (void)ge::AttrUtils::SetListInt(node->GetOpDescBarePtr(), ATTR_NAME_OUTPUT_MEM_TYPE_LIST, mem_type);
    }
  }
}

void SetEngine(ComputeGraphPtr &graph, const std::vector<string> &nodes, const string &engine_name) {
  for (auto node : graph->GetAllNodes()) {
    if (std::find(nodes.cbegin(), nodes.cend(), node->GetName()) != nodes.cend()) {
      node->GetOpDesc()->SetOpKernelLibName(engine_name);
    }
  }
}

void SetRefSrcVarName(ComputeGraphPtr &root_graph, const std::string &node_name, const string &src_node) {
  for (auto &node : root_graph->GetAllNodes()) {
    if (node->GetName() == node_name) {
      (void)ge::AttrUtils::SetStr(node->GetOpDesc()->MutableOutputDesc(0), REF_VAR_SRC_VAR_NAME, src_node);
    }
  }
}
}
void MemConflictShareGraph::SetOutReuseInput(ComputeGraphPtr &root_graph, const std::string &node_name) {
  for (auto &node : root_graph->GetAllNodes()) {
    if (node->GetName() == node_name) {
      (void)ge::AttrUtils::SetBool(node->GetOpDesc(), ATTR_NAME_OUTPUT_REUSE_INPUT, true);
      (void)ge::AttrUtils::SetInt(node->GetOpDesc(), ge::ATTR_NAME_REUSE_INPUT_ON_DIM_INDEX, 0);

      // 为了输出能和输入同符号
      auto output_tensor = node->GetOpDesc()->MutableOutputDesc(0);
      if (output_tensor != nullptr) {
        TensorUtils::SetReuseInput(*output_tensor, true);
        TensorUtils::SetReuseInputIndex(*output_tensor, 0U);
      }
    }
  }
}
void MemConflictShareGraph::SetNoPaddingContinuousInput(ComputeGraphPtr &root_graph, const std::string &node_name) {
  for (auto &node : root_graph->GetAllNodes()) {
    if (node->GetName() == node_name) {
      (void)ge::AttrUtils::SetBool(node->GetOpDesc(), ATTR_NAME_NOPADDING_CONTINUOUS_INPUT, true);
      (void)ge::AttrUtils::SetBool(node->GetOpDesc(), ATTR_NAME_OUTPUT_REUSE_INPUT, true);
      (void)ge::AttrUtils::SetInt(node->GetOpDesc(), ge::ATTR_NAME_REUSE_INPUT_ON_DIM_INDEX, 0);

      // 为了输出能和输入同符号
      auto output_tensor = node->GetOpDesc()->MutableOutputDesc(0);
      if (output_tensor != nullptr) {
        TensorUtils::SetReuseInput(*output_tensor, true);
        TensorUtils::SetReuseInputIndex(*output_tensor, 0U);
      }
    }
  }
}

void MemConflictShareGraph::SetNoPaddingContinuousOutput(ComputeGraphPtr &root_graph, const std::string &node_name) {
  for (auto &node : root_graph->GetAllNodes()) {
    if (node->GetName() == node_name) {
      (void)ge::AttrUtils::SetBool(node->GetOpDesc(), ATTR_NAME_NOPADDING_CONTINUOUS_OUTPUT, true);
      (void)ge::AttrUtils::SetBool(node->GetOpDesc(), ATTR_NAME_OUTPUT_REUSE_INPUT, true);
      (void)ge::AttrUtils::SetInt(node->GetOpDesc(), ge::ATTR_NAME_REUSE_INPUT_ON_DIM_INDEX, 0);

      // 为了输出能和输入同符号
      auto output_tensor = node->GetOpDesc()->MutableOutputDesc(0);
      if (output_tensor != nullptr) {
        TensorUtils::SetReuseInput(*output_tensor, true);
        TensorUtils::SetReuseInputIndex(*output_tensor, 0U);
      }
    }
  }
}
void MemConflictShareGraph::SetContinuousInput(ComputeGraphPtr &root_graph, const std::string &node_name) {
  for (auto &node : root_graph->GetAllNodes()) {
    if (node->GetName() == node_name) {
      (void)ge::AttrUtils::SetBool(node->GetOpDesc(), ATTR_NAME_CONTINUOUS_INPUT, true);
    }
  }
}
void MemConflictShareGraph::SetContinuousOutput(ComputeGraphPtr &root_graph, const std::string &node_name) {
  for (auto &node : root_graph->GetAllNodes()) {
    if (node->GetName() == node_name) {
      (void)ge::AttrUtils::SetBool(node->GetOpDesc(), ATTR_NAME_CONTINUOUS_OUTPUT, true);
    }
  }
}

void MemConflictShareGraph::SetShapeForAllNodes(ComputeGraphPtr &graph, const std::vector<int64_t> &shape) {
  for (auto &node : graph->GetAllNodes()) {
    for (size_t i = 0U; i < node->GetOutDataNodesSize(); ++i) {
      auto out_tensor = node->GetOpDescBarePtr()->MutableOutputDesc(i);
      out_tensor->SetShape(GeShape{shape});
    }

    for (size_t i = 0U; i < node->GetInDataNodesSize(); ++i) {
      auto in_tensor = node->GetOpDescBarePtr()->MutableInputDesc(i);
      in_tensor->SetShape(GeShape{shape});
    }
  }
}

void MemConflictShareGraph::SetShapeForNodesInputs(ComputeGraphPtr &graph, const std::vector<int64_t> &shape,
                                                   const std::vector<std::string> &names) {
  for (auto &node : graph->GetAllNodes()) {
    const auto node_name = node->GetName();
    const auto iter = std::find(names.begin(), names.end(), node_name);
    if (iter == names.end()) {
      continue;
    }
    for (size_t i = 0U; i < node->GetInDataNodesSize(); ++i) {
      auto in_tensor = node->GetOpDescBarePtr()->MutableInputDesc(i);
      in_tensor->SetShape(GeShape{shape});
    }
  }
}

void MemConflictShareGraph::SetShapeForNodesOutputs(ComputeGraphPtr &graph, const std::vector<int64_t> &shape,
                                                   const std::vector<std::string> &names) {
  for (auto &node : graph->GetAllNodes()) {
    const auto node_name = node->GetName();
    const auto iter = std::find(names.begin(), names.end(), node_name);
    if (iter == names.end()) {
      continue;
    }
    for (size_t i = 0U; i < node->GetOutDataNodesSize(); ++i) {
      auto out_tensor = node->GetOpDescBarePtr()->MutableOutputDesc(i);
      out_tensor->SetShape(GeShape{shape});
    }
  }
}

void MemConflictShareGraph::SetStreamForNodes(ComputeGraphPtr &graph, int64_t stream_id,
                                              const std::vector<std::string> &names) {
  for (auto &node : graph->GetAllNodes()) {
    const auto node_name = node->GetName();
    const auto iter = std::find(names.begin(), names.end(), node_name);
    if (iter == names.end()) {
      continue;
    }
    node->GetOpDescBarePtr()->SetStreamId(stream_id);
  }
}

Status EnsureOrder(std::vector<NodePtr> &origin_nodes, NodePtr &a, NodePtr &b) {
  auto a_iter = std::find(origin_nodes.begin(), origin_nodes.end(), a);
  auto b_iter = std::find(origin_nodes.begin(), origin_nodes.end(), b);
  if (a_iter == origin_nodes.end() || b_iter == origin_nodes.end()) {
    std::cerr << "EnsureOrder failed, can not find a or b"  << std::endl;
    return FAILED;
  }
  if (a_iter < b_iter) {
    return SUCCESS;
  }
  origin_nodes.erase(a_iter);
  origin_nodes.insert(b_iter, a);
  return SUCCESS;
}

Status MemConflictShareGraph::TopologicalSortingMock(const ComputeGraphPtr &graph,
                                                     const std::vector<std::string> &node_names) {
  graph->TopologicalSorting();
  std::vector<NodePtr> origin_nodes;
  for (const auto &node : graph->GetAllNodes()) {
    origin_nodes.emplace_back(node);
  }
  std::vector<NodePtr> param_nodes;
  for (size_t i = 0U; i < node_names.size(); ++i) {
    auto node = graph->FindNode(node_names.at(i));
    if (node == nullptr) {
      std::cerr << "TopologicalSortingMock failed, because can not find node: " << node_names.at(i) << std::endl;
      return FAILED;
    }
    param_nodes.emplace_back(node);
  }

  for (size_t i = 0U; i < param_nodes.size(); ++i) {
    auto cur_node = param_nodes[i];
    for (size_t j = i + 1U; j < param_nodes.size(); ++j) {
      EnsureOrder(origin_nodes, cur_node, param_nodes[j]);
    }
  }
  for (int64_t i = 0; i < origin_nodes.size(); ++i) {
    origin_nodes.at(i)->GetOpDesc()->SetId(i);
  }
  graph->ReorderByNodeId();

  return SUCCESS;
}
void MemConflictShareGraph::SetSizeForAllNodes(ComputeGraphPtr &graph) {
  for (auto &node : graph->GetAllNodes()) {
    for (size_t i = 0U; i < node->GetOutDataNodesSize(); ++i) {
      auto out_tensor = node->GetOpDescBarePtr()->MutableOutputDesc(i);
      int64_t tensor_size = 0;
      TensorUtils::CalcTensorMemSize(out_tensor->GetShape(), out_tensor->GetFormat(), out_tensor->GetDataType(), tensor_size);
      tensor_size = (tensor_size + 32 - 1) / 32 * 32 + 32;
      TensorUtils::SetSize(*out_tensor, tensor_size);
    }

    for (size_t i = 0U; i < node->GetInDataNodesSize(); ++i) {
      auto in_tensor = node->GetOpDescBarePtr()->MutableInputDesc(i);
      int64_t tensor_size = 0;
      TensorUtils::CalcTensorMemSize(in_tensor->GetShape(), in_tensor->GetFormat(), in_tensor->GetDataType(), tensor_size);
      tensor_size = (tensor_size + 32 - 1) / 32 * 32 + 32;
      TensorUtils::SetSize(*in_tensor, tensor_size);
    }
  }
}
/*
 *  data1  data2
 *     \    /
 *       if
 *       |
 *      op3
 *       |
 *    netoutput
 *
 * then_subgraph          else_subgraph
 * +------------------+   +------------------+
 * | data3--netoutput1|   | data5  netoutput2|
 * | data4            |   | data6-+/         |
 * +------------------+   +------------------+
 *
 */
ComputeGraphPtr MemConflictShareGraph::BuildUserInAndUserInGraph() {
  const auto data3 = OP_CFG(DATA).ParentNodeIndex(0);
  const auto data4 = OP_CFG(DATA).ParentNodeIndex(1);
  DEF_GRAPH(then_sub) {
                        CHAIN(NODE("data3", data3)->NODE("netoutput1", NETOUTPUT));
                        CHAIN(NODE("data4", data4));
                      };
  const auto data5 = OP_CFG(DATA).ParentNodeIndex(0);
  const auto data6 = OP_CFG(DATA).ParentNodeIndex(1);
  DEF_GRAPH(else_sub) {
                        CHAIN(NODE("data5", data5));
                        CHAIN(NODE("data6", data6)->NODE("netoutput2", NETOUTPUT));
                      };

  DEF_GRAPH(g1) {
                  CHAIN(NODE("data1", DATA)->EDGE(0, 0)->NODE("if", IF, then_sub, else_sub)
                            ->EDGE(0, 0)->NODE("op3", ADD)->EDGE(0, 0)->NODE("netoutput0", NETOUTPUT));
                  CHAIN(NODE("data2", DATA)->EDGE(0, 1)->NODE("if", IF, then_sub, else_sub));
                };
  auto graph = ToComputeGraph(g1);
  AddParentIndexForSubGraphNetoutput(graph);
  return graph;
}

/*
 *      data
 *       |
 *     netoutput
 */
ComputeGraphPtr MemConflictShareGraph::BuildUserInAndUserOutGraph() {
  DEF_GRAPH(g1) {
                  CHAIN(NODE("data", DATA)->NODE("netoutput", NETOUTPUT));
                };
  return ToComputeGraph(g1);
}

/*
 * unknown graph
 *      op1
 *       |               known sub graph
 * partitioned_call      +-----------+
 *       |               |   data    |
 *      op2              |    |      |
 *                       | netoutput |
 *                       +-----------+
 */
ComputeGraphPtr MemConflictShareGraph::BuildUserInAndUserOutGraphInSubGraph() {
  const auto sub_data = OP_CFG(DATA).ParentNodeIndex(0);
  DEF_GRAPH(sub_1) {
                     CHAIN(NODE("data", sub_data)->NODE("netoutput", NETOUTPUT));
                   };
  DEF_GRAPH(g1) {
                  CHAIN(NODE("op1", ADD)->NODE("partitioned_call", PARTITIONEDCALL, sub_1)->NODE("op2", ADD));
                };
  auto graph = ToComputeGraph(g1);
  graph->SetGraphUnknownFlag(true);
  return graph;
}

/*
 *
 * know graph
 *      data1
 *       |               known sub graph
 * partitioned_call      +-----------+
 *       |               |   data2   |
 *      op2              |    |      |
 *                       | netoutput |
 *                       +-----------+
 */
ComputeGraphPtr MemConflictShareGraph::BuildUserInAndNormalInWithSubGraph() {
  const auto sub_data = OP_CFG(DATA).ParentNodeIndex(0);
  DEF_GRAPH(sub_1) {
                     CHAIN(NODE("data2", sub_data)->NODE("netoutput", NETOUTPUT));
                   };
  DEF_GRAPH(g1) {
                  CHAIN(NODE("data1", DATA)->NODE("partitioned_call", PARTITIONEDCALL, sub_1)->NODE("op2", ADD));
                };
  auto graph = ToComputeGraph(g1);
  return graph;
}

/*
 *
 * know graph
 *      data1
 *       |               known sub graph
 * partitioned_call      +-----------+
 *       |               |   data2   |
 *      op2              |    |      |
 *                       |   add     |
 *                       |    |      |
 *                       | netoutput |
 *                       +-----------+
 */
ComputeGraphPtr MemConflictShareGraph::BuildUserInAndNormalOutWithSubGraph() {
  const auto sub_data = OP_CFG(DATA).ParentNodeIndex(0);
  DEF_GRAPH(sub_1) {
                     CHAIN(NODE("data2", sub_data)->NODE("add", ADD)->NODE("netoutput", NETOUTPUT));
                   };
  DEF_GRAPH(g1) {
                  CHAIN(NODE("data1", DATA)->NODE("partitioned_call", PARTITIONEDCALL, sub_1)->NODE("op2", ADD));
                };
  auto graph = ToComputeGraph(g1);
  return graph;
}

/*
 *
 * unknown graph
 *      op1
 *       |               known sub graph
 * partitioned_call      +-----------+
 *       |               |   data    |
 *      op2              |    |      |
 *                       |  switch   |
 *                       |    |      |
 *                       |   op3     |
 *                       |    |      |
 *                       | netoutput |
 *                       +-----------+
 */
ComputeGraphPtr MemConflictShareGraph::BuildUserInAndNotSupportRefreshInGraph() {
  const auto sub_data = OP_CFG(DATA).ParentNodeIndex(0);
  DEF_GRAPH(sub_1) {
                     CHAIN(NODE("data", sub_data)->NODE("switch", STREAMSWITCH)->NODE("op3", ADD)
                               ->NODE("netoutput", NETOUTPUT));
                   };
  DEF_GRAPH(g1) {
                  CHAIN(NODE("op1", ADD)->NODE("partitioned_call", PARTITIONEDCALL, sub_1)->NODE("op2", ADD));
                };
  auto graph = ToComputeGraph(g1);
  graph->SetGraphUnknownFlag(true);
  return graph;
};
/*
 *   data
 *    |
 *   swt
 *    |
 *    a
 *    |
 *  netoutput
 */
ComputeGraphPtr MemConflictShareGraph::BuildUserInAndNotSupportRefreshOutByRefGraph() {
  DEF_GRAPH(g1) {
                  CHAIN(NODE("data", DATA)->NODE("swt", STREAMSWITCH)->NODE("a", RELU)->NODE("netoutput", NETOUTPUT));
                };
  auto graph = ToComputeGraph(g1);
  SetOutReuseInput(graph, "swt");
  return graph;
};
/*
 *   data
 *    |
 *   hcom
 *    |
 *    a
 *    |
 *  netoutput
 */
ComputeGraphPtr MemConflictShareGraph::BuildUserInAndNotSupportRefreshOutByRefWithHcomGraph() {
  DEF_GRAPH(g1) {
                  CHAIN(NODE("data", DATA)->NODE("hcom", HCOMBROADCAST)->NODE("a", RELU)->NODE("netoutput", NETOUTPUT));
                };
  auto graph = ToComputeGraph(g1);
  SetEngine(graph, {"hcom"}, kEngineNameHccl);
  SetOutReuseInput(graph, "hcom");
  return graph;
};
/*
 *   data
 *    |
 *   swt1
 *    |
 *   swt2
 *    |
 *    a
 *    |
 *  netoutput
 */
ComputeGraphPtr MemConflictShareGraph::BuildUserInAndNotSupportRefreshOutByRefGraph2() {
  DEF_GRAPH(g1) {
                  CHAIN(NODE("data", DATA)->NODE("swt1", STREAMSWITCH)->
                  NODE("swt2", STREAMSWITCH)->NODE("a", RELU)->NODE("netoutput", NETOUTPUT));
                };
  auto graph = ToComputeGraph(g1);
  return graph;
};
/*
 *    data    swt
 *      \     /
 *     phonyconcat
 */
ComputeGraphPtr MemConflictShareGraph::BuildUserInAndNotSupportRefreshOutByContinuousInGraph() {
  DEF_GRAPH(g1) {
                  CHAIN(NODE("data", DATA)->NODE("phonyconcat", PHONYCONCAT));
                  CHAIN(NODE("swt", STREAMSWITCH)->NODE("phonyconcat", PHONYCONCAT));
                };
  auto graph = ToComputeGraph(g1);
  SetNoPaddingContinuousInput(graph, "phonyconcat");
  return graph;
};

/*
 *      data
 *       |
 *      op1 need p2p input
 */
ComputeGraphPtr MemConflictShareGraph::BuildUserInAndRtsSpecialInGraph() {
  DEF_GRAPH(g1) {
                  CHAIN(NODE("data", DATA)->NODE("op1", HCOMALLREDUCE));
                };
  std::vector<int64_t> in_mem_type {RT_MEMORY_P2P_DDR};
  auto graph = ToComputeGraph(g1);
  auto op1 = graph->FindNode("op1");
  (void)ge::AttrUtils::SetListInt(op1->GetOpDescBarePtr(), ATTR_NAME_INPUT_MEM_TYPE_LIST,
                                  in_mem_type);
  return graph;
}
/*
 *   refdata const
 *       \   /
 *      assign
 *       |
 *      op1 need p2p input
 */
ComputeGraphPtr MemConflictShareGraph::BuildUserInRefDataAndRtsSpecialInGraph() {
  vector<int64_t> perm1(1 * 1 * 224 * 224, 0);
  GeTensorDesc tensor_desc1(GeShape(vector<int64_t>{1, 1, 224, 224}));
  GeTensorPtr const_tensor1 =
      std::make_shared<GeTensor>(tensor_desc1, reinterpret_cast<uint8_t *>(perm1.data()) , sizeof(int64_t)*perm1.size());
  auto constant = OP_CFG(CONSTANT).Weight(const_tensor1);
  const auto assign = OP_CFG(ASSIGN).Attr(ATTR_NAME_REFERENCE, true).InNames({"ref"}).OutNames({"ref"});

  DEF_GRAPH(g1) {
    CHAIN(NODE("refdata", REFDATA)->NODE("assign", assign)
              ->NODE("op1", RELU)->NODE("netoutput", NETOUTPUT));
    CHAIN(NODE("const", constant)->NODE("assign", assign));
  };
  auto graph = ToComputeGraph(g1);
  SetRefSrcVarName(graph, "assign", "refdata");

  std::vector<int64_t> in_mem_type {RT_MEMORY_P2P_DDR};
  auto op1 = graph->FindNode("op1");
  (void)ge::AttrUtils::SetListInt(op1->GetOpDescBarePtr(), ATTR_NAME_INPUT_MEM_TYPE_LIST,
                                  in_mem_type);

  // 如果不设置，会被设置为动态shape图
  AttrUtils::SetBool(graph, ATTR_NAME_NO_NEED_DYNAMIC_SHAPE_PARTITION, true);

  return graph;
}

/*
 *   refdata const
 *       \   /
 *      assign
 *       |
 *      hcom need p2p output
 */
ComputeGraphPtr MemConflictShareGraph::BuildUserInRefDataAndRtsSpecialOutGraph() {
  vector<int64_t> perm1(1 * 1 * 224 * 224, 0);
  GeTensorDesc tensor_desc1(GeShape(vector<int64_t>{1, 1, 224, 224}));
  GeTensorPtr const_tensor1 =
      std::make_shared<GeTensor>(tensor_desc1, reinterpret_cast<uint8_t *>(perm1.data()) , sizeof(int64_t)*perm1.size());
  auto constant = OP_CFG(CONSTANT).Weight(const_tensor1);
  const auto assign = OP_CFG(ASSIGN).Attr(ATTR_NAME_REFERENCE, true).InNames({"ref"}).OutNames({"ref"});

  DEF_GRAPH(g1) {
    CHAIN(NODE("refdata", REFDATA)->NODE("assign", assign)
              ->NODE("hcom", HCOMBROADCAST)->NODE("netoutput", NETOUTPUT));
    CHAIN(NODE("const", constant)->NODE("assign", assign));
  };
  auto graph = ToComputeGraph(g1);
  SetRefSrcVarName(graph, "assign", "refdata");

  std::vector<int64_t> out_mem_type {RT_MEMORY_P2P_DDR};
  SetOutReuseInput(graph, "hcom");
  auto op1 = graph->FindNode("hcom");
  (void)ge::AttrUtils::SetListInt(op1->GetOpDescBarePtr(), ATTR_NAME_OUTPUT_MEM_TYPE_LIST,
                                  out_mem_type);
  // 如果不设置，会被设置为动态shape图
  AttrUtils::SetBool(graph, ATTR_NAME_NO_NEED_DYNAMIC_SHAPE_PARTITION, true);

  return graph;
}
/*
 *   refdata const
 *       \   /
 *      assign
 *       |
 *      swt
 */
ComputeGraphPtr MemConflictShareGraph::BuildUserInRefDataAndNotSupportRefreshInGraph() {
  vector<int64_t> perm1(1 * 1 * 224 * 224, 0);
  GeTensorDesc tensor_desc1(GeShape(vector<int64_t>{1, 1, 224, 224}));
  GeTensorPtr const_tensor1 =
      std::make_shared<GeTensor>(tensor_desc1, reinterpret_cast<uint8_t *>(perm1.data()) , sizeof(int64_t)*perm1.size());
  auto constant = OP_CFG(CONSTANT).Weight(const_tensor1);
  const auto assign = OP_CFG(ASSIGN).Attr(ATTR_NAME_REFERENCE, true).InNames({"ref"}).OutNames({"ref"});

  DEF_GRAPH(g1) {
    CHAIN(NODE("refdata", REFDATA)->NODE("assign", assign)
              ->NODE("swt", STREAMSWITCH)->NODE("netoutput", NETOUTPUT));
    CHAIN(NODE("const", constant)->NODE("assign", assign));
  };
  auto graph = ToComputeGraph(g1);
  SetRefSrcVarName(graph, "assign", "refdata");

  // 如果不设置，会被设置为动态shape图
  AttrUtils::SetBool(graph, ATTR_NAME_NO_NEED_DYNAMIC_SHAPE_PARTITION, true);

  return graph;
}
/*
 *   refdata const
 *       \   /
 *      assign
 *       |
 *      swt
 */
ComputeGraphPtr MemConflictShareGraph::BuildUserInRefDataAndNotSupportRefreshOutGraph() {
  vector<int64_t> perm1(1 * 1 * 224 * 224, 0);
  GeTensorDesc tensor_desc1(GeShape(vector<int64_t>{1, 1, 224, 224}));
  GeTensorPtr const_tensor1 =
      std::make_shared<GeTensor>(tensor_desc1, reinterpret_cast<uint8_t *>(perm1.data()) , sizeof(int64_t)*perm1.size());
  auto constant = OP_CFG(CONSTANT).Weight(const_tensor1);
  const auto assign = OP_CFG(ASSIGN).Attr(ATTR_NAME_REFERENCE, true).InNames({"ref"}).OutNames({"ref"});

  DEF_GRAPH(g1) {
    CHAIN(NODE("refdata", REFDATA)->NODE("assign", assign)
              ->NODE("swt", STREAMSWITCH)->NODE("netoutput", NETOUTPUT));
    CHAIN(NODE("const", constant)->NODE("assign", assign));
  };
  auto graph = ToComputeGraph(g1);
  SetRefSrcVarName(graph, "assign", "refdata");

  SetOutReuseInput(graph, "swt");

  // 如果不设置，会被设置为动态shape图
  AttrUtils::SetBool(graph, ATTR_NAME_NO_NEED_DYNAMIC_SHAPE_PARTITION, true);

  return graph;
}
/*
 *      data
 *       |
 *      hcom (输出引用输入，输出需要p2p类型内存)
 *       |
 *       a
 */
ComputeGraphPtr MemConflictShareGraph::BuildUserInAndRtsSpecialOutGraph() {
  DEF_GRAPH(g1) {
                  CHAIN(NODE("data", DATA)->NODE("hcom", HCOMALLREDUCE)->NODE("a", RELU));
                };
  std::vector<int64_t> out_mem_type {RT_MEMORY_P2P_DDR};
  auto graph = ToComputeGraph(g1);
  SetOutReuseInput(graph, "hcom");
  auto op1 = graph->FindNode("hcom");
  (void)ge::AttrUtils::SetListInt(op1->GetOpDescBarePtr(), ATTR_NAME_OUTPUT_MEM_TYPE_LIST,
                                  out_mem_type);
  return graph;
}
/*
 *             data1    data2                                            data11   partitioned_call1 ┌──────────┐
 *                │       │                                                │         │              │constant1 │
 *                │       │                                                │         │              │    │     │
 *                │       │                                                │         │              │    │     │
 *          ┌─────┴──┐ ┌──┘                                         ┌──────┴───┐ ┌───┘              │netoutput5│
 *          │        │ │                                            │          │ │                  └──────────┘
 *          │        │ │                                            │          │ │
 * partitioned_call2 If1                                            swt        If2
 * ┌───────────┐      │  then_subgraph1     else_subgraph1          │           │   then_subgraph2     else_subgraph2
 * │ data12    │      │ ┌─────────────┐   ┌─────────────┐           │           │  ┌─────────────┐   ┌─────────────┐
 * │   ┼       │      │ │ data3  data4│   │ data5  data6│           │           │  │ data7  data8│   │ data9 data10│
 * │ netoutput6│      │ │   │         │   │  |       │  │           │           │  │   │         │   │          │  │
 * └───┬───────┘      │ │   └──┬      │   │ op5   ┬──┘  │           │           │  │   └──┬      │   │       ┬──┘  │
 *     │              │ │      │      │   │       │     │           │           │  │      │      │   │       │     │
 *     │              │ │   netoutput1│   │  netoutput2 │           │           │  │   netoutput3│   │  netoutput4 │
 *     │              │ └─────────────┘   └─────────────┘           │          op4 └─────────────┘   └─────────────┘
 *     │             op3                                            │           |
 *     │              │                                             │           │
 *     │              │                                             │           │
 *     │              └─────────────────────────────┐ ┌─────────────┘           │
 *     │                                            │ │                         │
 *     └──────────────────────────────────────────┐ │ │                         │
 *                                                │ │ │  +──────────────────────┘
 *                                                │ │ │  │
 *                                                netoutput8
 */
ComputeGraphPtr MemConflictShareGraph::BuildUserInAndOtherConflictTypeGraph() {
  const auto data3 = OP_CFG(DATA).ParentNodeIndex(0);
  const auto data4 = OP_CFG(DATA).ParentNodeIndex(1);
  DEF_GRAPH(then_sub1) {
                        CHAIN(NODE("data3", data3)->NODE("netoutput1", NETOUTPUT));
                        CHAIN(NODE("data4", data4));
                      };
  const auto data5 = OP_CFG(DATA).ParentNodeIndex(0);
  const auto data6 = OP_CFG(DATA).ParentNodeIndex(1);
  DEF_GRAPH(else_sub1) {
                        CHAIN(NODE("data5", data5)->NODE("op5", MUL)->Ctrl()->NODE("netoutput2", NETOUTPUT));
                        CHAIN(NODE("data6", data6)->NODE("netoutput2", NETOUTPUT));
                        CHAIN(NODE("data6", data6)->EDGE(0, 1)->NODE("op5", MUL));
                      };

  const auto data7 = OP_CFG(DATA).ParentNodeIndex(0);
  const auto data8 = OP_CFG(DATA).ParentNodeIndex(1);
  DEF_GRAPH(then_sub2) {
                         CHAIN(NODE("data7", data7)->NODE("netoutput3", NETOUTPUT));
                         CHAIN(NODE("data8", data8));
                       };
  const auto data9 = OP_CFG(DATA).ParentNodeIndex(0);
  const auto data10 = OP_CFG(DATA).ParentNodeIndex(1);
  DEF_GRAPH(else_sub2) {
                         CHAIN(NODE("data9", data9));
                         CHAIN(NODE("data10", data10)->NODE("netoutput4", NETOUTPUT));
                       };

  vector<int64_t> perm1(1 * 1 * 224 * 224, 0);
  GeTensorDesc tensor_desc1(GeShape(vector<int64_t>{1, 1, 224, 224}));
  GeTensorPtr const_tensor1 =
      std::make_shared<GeTensor>(tensor_desc1, reinterpret_cast<uint8_t *>(perm1.data()) , sizeof(int64_t)*perm1.size());
  auto constant1 = OP_CFG(CONSTANTOP).Weight(const_tensor1);

  DEF_GRAPH(partitioned_call1) {
                         CHAIN(NODE("constant1", constant1)->NODE("netoutput5", NETOUTPUT));
                       };

  const auto data12 = OP_CFG(DATA).ParentNodeIndex(0);
  DEF_GRAPH(partitioned_call2) {
                                 CHAIN(NODE("data12", data12)->NODE("netoutput6", NETOUTPUT));
                               };

  DEF_GRAPH(g1) {
                  CHAIN(NODE("data1", DATA)->EDGE(0, 0)->NODE("partitioned_call2", PARTITIONEDCALL, partitioned_call2)
                            ->EDGE(0, 0)->NODE("netoutput8", NETOUTPUT));
                  CHAIN(NODE("data1", DATA)->EDGE(0, 0)->NODE("if1", IF, then_sub1, else_sub1)
                            ->EDGE(0, 0)->NODE("op3", TRANSDATA)->EDGE(0, 1)->NODE("netoutput8", NETOUTPUT));
                  CHAIN(NODE("data2", DATA)->EDGE(0, 1)->NODE("if1", IF, then_sub1, else_sub1));

                  CHAIN(NODE("data11", DATA)->EDGE(0, 0)->NODE("swt", LABELSWITCHBYINDEX)
                            ->EDGE(0, 2)->NODE("netoutput8", NETOUTPUT));
                  CHAIN(NODE("data11", DATA)->EDGE(0, 0)->NODE("if2", IF, then_sub2, else_sub2)
                            ->EDGE(0, 0)->NODE("op4", TRANSDATA)->EDGE(0, 3)->NODE("netoutput8", NETOUTPUT));
                  CHAIN(NODE("partitioned_call1", PARTITIONEDCALL, partitioned_call1)
                            ->EDGE(0, 1)->NODE("if2", IF, then_sub2, else_sub2));
                };

  auto graph = ToComputeGraph(g1);
  AddParentIndexForSubGraphNetoutput(graph);
  // 如果不设置，会被设置为动态shape图
  AttrUtils::SetBool(graph, ATTR_NAME_NO_NEED_DYNAMIC_SHAPE_PARTITION, true);

  auto data11_node = graph->FindNode("data11");
  auto shape = data11_node->GetOpDescBarePtr()->GetOutputDesc(0).GetShape();
  for (auto node : graph->GetAllNodesPtr()) {
    for (size_t i = 0U; i < node->GetOpDescBarePtr()->GetOutputsSize(); ++i) {
      node->GetOpDescBarePtr()->MutableOutputDesc(i)->SetShape(shape);
    }
    for (size_t i = 0U; i < node->GetOpDescBarePtr()->GetInputsSize(); ++i) {
      node->GetOpDescBarePtr()->MutableInputDesc(i)->SetShape(shape);
    }
  }
  return graph;
}

/*
 *             data1    data2
 *                │       │
 *                │       │
 *                │       │
 *          ┌─────┴──┐ ┌──┘
 *          │        │ │ +---op0
 *          │        │ │ |0
 * partitioned_call2  If1
 * ┌───────────┐      │  then_subgraph1     else_subgraph1
 * │ data7     │      │ ┌─────────────┐   ┌─────────────┐
 * │   ┼       │      │ │ data3  data4│   │ data5  data6│
 * │ netoutput3│      │ │   │         │   │  |       │  │
 * └───┬───────┘      │ │   └──┬      │   │ op2   ┬──┘  │
 *     │              │ │      │      │   │       │     │
 *     │              │ │   netoutput1│   │  netoutput2 │
 *     │              │ └─────────────┘   └─────────────┘
 *     │             op1
 *     │              |
 *     +-----+  +-----+
 *           |  |
 *         netoutput4
 */
ComputeGraphPtr MemConflictShareGraph::BuildUserInAndUserIO() {
  const auto data3 = OP_CFG(DATA).ParentNodeIndex(1);
  const auto data4 = OP_CFG(DATA).ParentNodeIndex(2);
  DEF_GRAPH(then_sub1) {
                         CHAIN(NODE("data3", data3)->NODE("netoutput1", NETOUTPUT));
                         CHAIN(NODE("data4", data4));
                       };
  const auto data5 = OP_CFG(DATA).ParentNodeIndex(1);
  const auto data6 = OP_CFG(DATA).ParentNodeIndex(2);
  DEF_GRAPH(else_sub1) {
                         CHAIN(NODE("data5", data5)->NODE("op2", MUL)->Ctrl()->NODE("netoutput2", NETOUTPUT));
                         CHAIN(NODE("data6", data6)->NODE("netoutput2", NETOUTPUT));
                         CHAIN(NODE("data6", data6)->EDGE(0, 1)->NODE("op2", MUL));
                       };

  const auto data7 = OP_CFG(DATA).ParentNodeIndex(0);
  DEF_GRAPH(partitioned_call2) {
                                 CHAIN(NODE("data7", data7)->NODE("netoutput3", NETOUTPUT));
                               };

  DEF_GRAPH(g1) {
                  CHAIN(NODE("data1", DATA)->EDGE(0, 0)->NODE("partitioned_call2", PARTITIONEDCALL, partitioned_call2)
                            ->EDGE(0, 0)->NODE("netoutput4", NETOUTPUT));
                  CHAIN(NODE("data1", DATA)->EDGE(0, 1)->NODE("if1", IF, then_sub1, else_sub1)
                            ->EDGE(0, 0)->NODE("op1", TRANSDATA)->EDGE(0, 1)->NODE("netoutput4", NETOUTPUT));
                  CHAIN(NODE("data2", DATA)->EDGE(0, 2)->NODE("if1", IF, then_sub1, else_sub1));
                  CHAIN(NODE("op0", RELU)->EDGE(0, 0)->NODE("if1", IF, then_sub1, else_sub1));
                };

  auto graph = ToComputeGraph(g1);
  AddParentIndexForSubGraphNetoutput(graph);
  // 如果不设置，会被设置为动态shape图
  AttrUtils::SetBool(graph, ATTR_NAME_NO_NEED_DYNAMIC_SHAPE_PARTITION, true);
  return graph;
}

/*
 *          data1
 *            │
 *            │
 *            │    op1
 *     ┌──────┴───┐ |
 *     │          │ |
 *     │          │ |0
 *    swt         If1
 *     │           │   then_subgraph1   else_subgraph1
 *     │           │  ┌─────────────┐   ┌──────────---───┐
 *     │           │  │ data2       │   │data3 constant1 │
 *     │           │  │   │         │   │        │       │
 *     │           │  │   └──┬      │   │        │       │
 *     │           │  │      │      │   │        │       │
 *     │           │  │   netoutput1│   │    netoutput2  │
 *     │          op2 └─────────────┘   └──────────---───┘
 *     │           |
 *     +--+  +-----+
 *        |  |
 *      netoutput
 */
ComputeGraphPtr MemConflictShareGraph::BuildUserInAndUnRefreshableInputAndConstInput() {
  const auto data2 = OP_CFG(DATA).ParentNodeIndex(1);
  DEF_GRAPH(then_sub1) {
                         CHAIN(NODE("data2", data2)->NODE("netoutput1", NETOUTPUT));
                       };

  const auto data3 = OP_CFG(DATA).ParentNodeIndex(1);
  vector<int64_t> perm1(1 * 1 * 224 * 224, 0);
  GeTensorDesc tensor_desc1(GeShape(vector<int64_t>{1, 1, 224, 224}));
  GeTensorPtr const_tensor1 =
      std::make_shared<GeTensor>(tensor_desc1, reinterpret_cast<uint8_t *>(perm1.data()) , sizeof(int64_t)*perm1.size());
  auto constant1 = OP_CFG(CONSTANTOP).Weight(const_tensor1);
  DEF_GRAPH(else_sub1) {
                         CHAIN(NODE("data3", data3));
                         CHAIN(NODE("constant1", constant1)->NODE("netoutput2", NETOUTPUT));
                       };

  DEF_GRAPH(g1) {
                  CHAIN(NODE("data1", DATA)->EDGE(0, 0)->NODE("swt", LABELSWITCHBYINDEX)
                            ->EDGE(0, 0)->NODE("netoutput", NETOUTPUT));
                  CHAIN(NODE("data1", DATA)->EDGE(0, 1)->NODE("if1", IF, then_sub1, else_sub1)
                            ->EDGE(0, 0)->NODE("op2", TRANSDATA)->EDGE(0, 1)->NODE("netoutput", NETOUTPUT));
                  CHAIN(NODE("op1", RELU)->EDGE(0, 0)->NODE("if1", IF, then_sub1, else_sub1));
                };

  auto graph = ToComputeGraph(g1);
  AddParentIndexForSubGraphNetoutput(graph);
  // 如果不设置，会被设置为动态shape图
  AttrUtils::SetBool(graph, ATTR_NAME_NO_NEED_DYNAMIC_SHAPE_PARTITION, true);

  return graph;
}

/*
 *   data--netoutput
 */
ComputeGraphPtr MemConflictShareGraph::BuildUserInConnectNetoutputGraph() {
  DEF_GRAPH(g1) {
                  CHAIN(NODE("data", DATA)->NODE("netoutput", NETOUTPUT));
                };
  auto graph = ToComputeGraph(g1);
  // 如果不设置，会被设置为动态shape图
  AttrUtils::SetBool(graph, ATTR_NAME_NO_NEED_DYNAMIC_SHAPE_PARTITION, true);
  return graph;
}

/*
 *    data1  data2
 *      \     /
 *        hcom (连续输入内存)
 *         |
 *         a
 */
ComputeGraphPtr MemConflictShareGraph::BuildUserInConnectContinuousInputGraph() {
  DEF_GRAPH(g1) {
                  CHAIN(NODE("data1", DATA)->NODE("hcom", HCOMALLGATHER)->NODE("a", ADD));
                  CHAIN(NODE("data2", DATA)->NODE("hcom", HCOMALLGATHER));
                };
  auto graph = ToComputeGraph(g1);
  // 如果不设置，会被设置为动态shape图
  AttrUtils::SetBool(graph, ATTR_NAME_NO_NEED_DYNAMIC_SHAPE_PARTITION, true);
  SetContinuousInput(graph, "hcom");
  return graph;
}
/*
 *  data1 data2
 *    \    /
 *     hcom (输出引用输入，且连续输出内存)
 *     / \
 *    a   b
 */
ComputeGraphPtr MemConflictShareGraph::BuildUserInConnectContinuousOutputGraph() {
  const auto hcom = OP_CFG(HCOMBROADCAST).Attr(ATTR_NAME_REFERENCE, true).InNames({"x1", "x2"}).OutNames({"x1", "x2"});
  DEF_GRAPH(g1) {
                  CHAIN(NODE("data1", DATA)->NODE("hcom", hcom)->NODE("a", ADD));
                  CHAIN(NODE("data2", DATA)->NODE("hcom", hcom)->NODE("b", ADD));
                };
  auto graph = ToComputeGraph(g1);
  // 如果不设置，会被设置为动态shape图
  AttrUtils::SetBool(graph, ATTR_NAME_NO_NEED_DYNAMIC_SHAPE_PARTITION, true);
  SetContinuousOutput(graph, "hcom");
  return graph;
}

/*
 *    data1  data2
 *      \     /
 *     PhonyConcat (NoPadding连续输入内存)
 *         |
 *         a
 */
ComputeGraphPtr MemConflictShareGraph::BuildUserInConnectNoPaddingContinuousInputGraph() {
  DEF_GRAPH(g1) {
                  CHAIN(NODE("data1", DATA)->NODE("phony_concat", "PhonyConcat")->NODE("a", ADD));
                  CHAIN(NODE("data2", DATA)->NODE("phony_concat", "PhonyConcat"));
                };
  auto graph = ToComputeGraph(g1);
  // 如果不设置，会被设置为动态shape图
  AttrUtils::SetBool(graph, ATTR_NAME_NO_NEED_DYNAMIC_SHAPE_PARTITION, true);
  SetNoPaddingContinuousInput(graph, "phony_concat");
  return graph;
}
/*
 *     data
 *      |
 *   phony_split (输出引用输入，且NoPadding连续输出内存)
 *     /  \
 *    a    b
 */
ComputeGraphPtr MemConflictShareGraph::BuildUserInConnectNoPaddingContinuousOutputGraph() {
  const auto data = OP_CFG(DATA).TensorDesc(FORMAT_NHWC, DT_FLOAT, {1, 1, 224, 448});
  const auto phony_split = OP_CFG("PhonySplit");
  DEF_GRAPH(g1) {
                  CHAIN(NODE("data", data)->NODE("phony_split", phony_split)->NODE("a", ADD)->NODE("netoutput", NETOUTPUT));
                  CHAIN(NODE("phony_split", phony_split)->NODE("a", ADD));
                };
  auto graph = ToComputeGraph(g1);

  auto ps = graph->FindNode("phony_split");
  ps->GetOpDescBarePtr()->MutableOutputDesc(0)->SetShape(GeShape(std::vector<int64_t>({1, 1, 112, 448})));
  ps->GetOpDescBarePtr()->MutableOutputDesc(1)->SetShape(GeShape(std::vector<int64_t>({1, 1, 112, 448})));

  // 如果不设置，会被设置为动态shape图
  AttrUtils::SetBool(graph, ATTR_NAME_NO_NEED_DYNAMIC_SHAPE_PARTITION, true);
  SetNoPaddingContinuousOutput(graph, "phony_split");
  return graph;
}
/*
 *      data1 const1 var1 const2  data2 data4
 *        \      |    \    /        |   /
 *         \     |    assign       hcom2
 *          \    |     /           /   \
 *           \   |   /            /   partitioned_call  +---------+
 *             hcom1             /    /                 | data3   |
 *               \              /    /                  |   |     |
 *                \            /    /                   |   a     |
 *                  \        /    /                     |   |     |
 *                   \      /   /                       |netoutput1|
 *                    netoutput                         +---------+
 */
ComputeGraphPtr MemConflictShareGraph::BuildUserInOutConnectContinuousInAndOutGraph() {
  vector<int64_t> perm1(1 * 1 * 224 * 224, 0);
  GeTensorDesc tensor_desc1(GeShape(vector<int64_t>{1, 1, 224, 224}));
  GeTensorPtr const_tensor1 =
      std::make_shared<GeTensor>(tensor_desc1, reinterpret_cast<uint8_t *>(perm1.data()) , sizeof(int64_t)*perm1.size());
  auto const1 = OP_CFG(CONSTANTOP).Weight(const_tensor1);
  auto const2 = OP_CFG(CONSTANTOP).Weight(const_tensor1);

  const auto data3 = OP_CFG(DATA).ParentNodeIndex(0);
  DEF_GRAPH(partitioned_call) {
                                 CHAIN(NODE("data3", data3)->NODE("a", RELU)->NODE("netoutput1", NETOUTPUT));
                               };
  const auto assign = OP_CFG(ASSIGN).Attr(ATTR_NAME_REFERENCE, true).InNames({"ref"}).OutNames({"ref"});
  DEF_GRAPH(g1) {
                  CHAIN(NODE("data1", DATA)->NODE("hcom1", HCOMREDUCESCATTER)->NODE("netoutput", NETOUTPUT));
                  CHAIN(NODE("const1", const1)->NODE("hcom1", HCOMREDUCESCATTER));
                  CHAIN(NODE("var1", VARIABLE)->NODE("assign", assign)->NODE("hcom1", HCOMREDUCESCATTER));
                  CHAIN(NODE("const2", const2)->NODE("assign", assign));

                  CHAIN(NODE("data2", DATA)->NODE("hcom2", HCOMREDUCESCATTER)->NODE("netoutput", NETOUTPUT));
                  CHAIN(NODE("data4", DATA)->NODE("hcom2", HCOMREDUCESCATTER));
                  CHAIN(NODE("hcom2", HCOMREDUCESCATTER)->NODE("partitioned_call", PARTITIONEDCALL, partitioned_call)
                            ->NODE("netoutput", NETOUTPUT));
  };
  auto graph = ToComputeGraph(g1);
  AddParentIndexForSubGraphNetoutput(graph);
  SetContinuousInput(graph, "hcom1");
  SetContinuousOutput(graph, "hcom2");

  // hcom2 两个输出分别引用两个输入
  SetOutReuseInput(graph, "hcom2");
  auto hcom2_node = graph->FindNode("hcom2");
  auto output_tensor = hcom2_node->GetOpDesc()->MutableOutputDesc(1);
  if (output_tensor != nullptr) {
    TensorUtils::SetReuseInput(*output_tensor, true);
    TensorUtils::SetReuseInputIndex(*output_tensor, 1U);
  }
  hcom2_node->GetOpDescBarePtr()->MutableOutputDesc(0)->SetShape(GeShape(std::vector<int64_t>({1, 1, 224, 224})));
  hcom2_node->GetOpDescBarePtr()->MutableOutputDesc(1)->SetShape(GeShape(std::vector<int64_t>({1, 1, 224, 224})));
  auto hcom1_node = graph->FindNode("hcom1");
  hcom1_node->GetOpDescBarePtr()->MutableOutputDesc(0)->SetShape(GeShape(std::vector<int64_t>({1, 1, 224, 224})));

  // 如果不设置，会被设置为动态shape图
  AttrUtils::SetBool(graph, ATTR_NAME_NO_NEED_DYNAMIC_SHAPE_PARTITION, true);
  return graph;
}

/*
 *      data1 const1 var1 const2  data2
 *        \      |    \    /        |
 *         \     |    assign       phonysplit
 *          \    |     /           /   \
 *           \   |   /            /   partitioned_call  +---------+
 *          phonyconcat1         /    /                 | data3   |
 *               \              /    /                  |   |     |
 *                \            /    /                   |   a     |
 *                  \        /    /                     |   |     |
 *                   \      /   /                       |netoutput1|
 *                    netoutput                         +---------+
 */
ComputeGraphPtr MemConflictShareGraph::BuildUserInOutConnectNoPaddingContinuousInAndOutGraph() {
  vector<int64_t> perm1(1 * 1 * 224 * 224, 0);
  GeTensorDesc tensor_desc1(GeShape(vector<int64_t>{1, 1, 224, 224}));
  GeTensorPtr const_tensor1 =
      std::make_shared<GeTensor>(tensor_desc1, reinterpret_cast<uint8_t *>(perm1.data()) , sizeof(int64_t)*perm1.size());
  auto const1 = OP_CFG(CONSTANTOP).Weight(const_tensor1);
  auto const2 = OP_CFG(CONSTANTOP).Weight(const_tensor1);

  const auto data3 = OP_CFG(DATA).ParentNodeIndex(0);
  DEF_GRAPH(partitioned_call) {
                                CHAIN(NODE("data3", data3)->NODE("a", RELU)->NODE("netoutput1", NETOUTPUT));
                              };
  const auto assign = OP_CFG(ASSIGN).Attr(ATTR_NAME_REFERENCE, true).InNames({"ref"}).OutNames({"ref"});
  DEF_GRAPH(g1) {
                  CHAIN(NODE("data1", DATA)->NODE("phonyconcat1", PHONYCONCAT)->NODE("netoutput", NETOUTPUT));
                  CHAIN(NODE("const1", const1)->NODE("phonyconcat1", PHONYCONCAT));
                  CHAIN(NODE("var1", VARIABLE)->NODE("assign", assign)->NODE("phonyconcat1", PHONYCONCAT));
                  CHAIN(NODE("const2", const2)->NODE("assign", assign));

                  CHAIN(NODE("data2", DATA)->NODE("phonysplit", PHONYSPLIT)->NODE("netoutput", NETOUTPUT));
                  CHAIN(NODE("phonysplit", PHONYSPLIT)->NODE("partitioned_call", PARTITIONEDCALL, partitioned_call)
                            ->NODE("netoutput", NETOUTPUT));
                };
  auto graph = ToComputeGraph(g1);
  AddParentIndexForSubGraphNetoutput(graph);
  SetNoPaddingContinuousInput(graph, "phonyconcat1");
  auto phonyconcat1_node = graph->FindNode("phonyconcat1");
  phonyconcat1_node->GetOpDescBarePtr()->MutableOutputDesc(0)->SetShape(GeShape(vector<int64_t>{1, 3, 224, 224}));
  /*
   * 由于其他用例中会设置PhonyConcat的引擎or注册optimizer?，导致会走到 GeLocalGraphOptimizer::OptimizeOriginalGraphJudgeInsert，
   * 里面会对PhonyConcat设置Nopadding连续输入,单独跑一个用例，和全量跑所有ST用例可能不一致.
   * 如果想测试NoPadding连续输出，就不要用PhonyConcat。
   */
  SetNoPaddingContinuousOutput(graph, "phonysplit");

  // phonysplit 两个输出分别引用两个输入
  auto ps = graph->FindNode("phonysplit");
  ps->GetOpDescBarePtr()->MutableOutputDesc(0)->SetShape(GeShape(std::vector<int64_t>({1, 1, 112, 224})));
  ps->GetOpDescBarePtr()->MutableOutputDesc(1)->SetShape(GeShape(std::vector<int64_t>({1, 1, 112, 224})));

  // 如果不设置，会被设置为动态shape图
  AttrUtils::SetBool(graph, ATTR_NAME_NO_NEED_DYNAMIC_SHAPE_PARTITION, true);
  return graph;
}
/*
 *   refdata  const
 *     \    /
 *     assign   a
 *        \    /
 *          pc
 *           |
 *           b
 *           |
 *       netoutput
 */
ComputeGraphPtr MemConflictShareGraph::BuildUserInAndNoPaddingContinuousInByAssignOutput() {
  vector<int64_t> perm1(1 * 1 * 224 * 224, 0);
  GeTensorDesc tensor_desc1(GeShape(vector<int64_t>{1, 1, 224, 224}));
  GeTensorPtr const_tensor1 =
      std::make_shared<GeTensor>(tensor_desc1, reinterpret_cast<uint8_t *>(perm1.data()) , sizeof(int64_t)*perm1.size());
  auto constant = OP_CFG(CONSTANT).Weight(const_tensor1);
  auto pc = OP_CFG(PHONYCONCAT);
  const auto assign = OP_CFG(ASSIGN).Attr(ATTR_NAME_REFERENCE, true).InNames({"ref"}).OutNames({"ref"});

  DEF_GRAPH(g1) {
    CHAIN(NODE("refdata", REFDATA)->NODE("assign", assign)->NODE("pc", pc)
              ->NODE("b", RELU)->NODE("netoutput", NETOUTPUT));
    CHAIN(NODE("const", constant)->NODE("assign", assign));
    CHAIN(NODE("a", RELU)->NODE("pc", pc)->NODE("netoutput", NETOUTPUT));
  };
  auto graph = ToComputeGraph(g1);
  SetRefSrcVarName(graph, "assign", "refdata");
  SetNoPaddingContinuousInput(graph, "pc");

  // 如果不设置，会被设置为动态shape图
  AttrUtils::SetBool(graph, ATTR_NAME_NO_NEED_DYNAMIC_SHAPE_PARTITION, true);
  return graph;
}


/*
 * constant---netoutput
 */
ComputeGraphPtr MemConflictShareGraph::BuildConstantConnectNetoutputGraph() {
  vector<float> perm1{0, 3, 1, 2};
  GeTensorDesc tensor_desc1(GeShape(vector<int64_t>{4}));
  GeTensorPtr const_tensor1 =
      std::make_shared<GeTensor>(tensor_desc1, reinterpret_cast<uint8_t *>(perm1.data()) ,
                                 sizeof(float) * perm1.size());
  auto constant1 = OP_CFG(CONSTANTOP).Weight(const_tensor1);

  DEF_GRAPH(g1) {
                  CHAIN(NODE("constant", constant1)->NODE("netoutput", NETOUTPUT));
                };
  auto graph = ToComputeGraph(g1);
  // 如果不设置，会被设置为动态shape图
  AttrUtils::SetBool(graph, ATTR_NAME_NO_NEED_DYNAMIC_SHAPE_PARTITION, true);
  return graph;
}

/*
 *   data1
 *     |
 *    op1
 *     |
 * partitioned_call  +----------------+
 *     |             |data2  constant |
 *    op2            |         |      |
 *     |             |      netoutput1|
 *  netoutput2       +----------------+
 */
ComputeGraphPtr MemConflictShareGraph::BuildConstantConnectNetoutputSubGraph() {
  const auto sub_data = OP_CFG(DATA).ParentNodeIndex(0);
  vector<float> perm1{0, 3, 1, 2};
  GeTensorDesc tensor_desc1(GeShape(vector<int64_t>{4}));
  GeTensorPtr const_tensor1 =
      std::make_shared<GeTensor>(tensor_desc1, reinterpret_cast<uint8_t *>(perm1.data()) ,
                                 sizeof(float) * perm1.size());
  auto constant1 = OP_CFG(CONSTANTOP).Weight(const_tensor1);
  DEF_GRAPH(sub_1) {
                     CHAIN(NODE("data2", sub_data));
                     CHAIN(NODE("constant", constant1)->NODE("netoutput1", NETOUTPUT));
                   };
  DEF_GRAPH(g1) {
                  CHAIN(NODE("data1", DATA)->NODE("op1", RELU)->NODE("partitioned_call", PARTITIONEDCALL, sub_1)
                            ->NODE("op2", RELU)->NODE("netoutput2", NETOUTPUT));
                };
  auto graph = ToComputeGraph(g1);
  AddParentIndexForSubGraphNetoutput(graph);
  // 如果不设置，会被设置为动态shape图
  AttrUtils::SetBool(graph, ATTR_NAME_NO_NEED_DYNAMIC_SHAPE_PARTITION, true);
  return graph;
}
/*
 *   op1 partitioned_call1
 *    |   | | |
 *    |   | | |  op2
 *   0\  1|2/3/4/
 *       if
 *        ┼
 *        |
 *       op5
 *        ┼
 *       netoutput
 *
 *
 *   partitioned_call1          then_sub                             else_sub
 *  ┌────────────────────┐    ┌───────────────────────────────┐   ┌───────────────────────────────────────┐
 *  │ var const1 refdata1│    │ data1 data2 data3 data4 data5 │   │  data6 data7 data8 data9 data10       │
 *  │  │     │      │    │    │         ┼     │     │     │   │   │                                       │
 *  │  └───┐ │  ┌───┘    │    │       refnode │     │     │   │   │   refdata2       op3                  │
 *  │      │ │  │        │    │         |     │     │     │   │   │      ┼            ┼                   │
 *  │    netoutput1      │    │         | +---+     │     │   │   │   transdata1    transdata2   swt  op4 │
 *  └────────────────────┘    │         + | +-------+     │   │   │ (ref refdata1)  (ref var)     │    │  │
 *                            │         | | | ┌───────────┘   │   │          │           │ ┌──────┘    │  │
 *                            │         ┼ ┼ ┼ │               │   │          └────────┐  │ │ +---------+  │
 *                            │        netoutput2             │   │                   │  │ │ |            │
 *                            └───────────────────────────────┘   │                 netoutput3            │
 *                                                                └───────────────────────────────────────┘
 */
ComputeGraphPtr MemConflictShareGraph::BuildImmutableOutAndOtherTypeGraph() {
  vector<int64_t> perm1(1 * 1 * 224 * 224, 0);
  GeTensorDesc tensor_desc1(GeShape(vector<int64_t>{1, 1, 224, 224}));
  GeTensorPtr const_tensor1 =
      std::make_shared<GeTensor>(tensor_desc1, reinterpret_cast<uint8_t *>(perm1.data()) , sizeof(int64_t)*perm1.size());
  auto const1 = OP_CFG(CONSTANTOP).Weight(const_tensor1);
  const auto refdata1 = OP_CFG(REFDATA).InCnt(1);
  const auto data1 = OP_CFG(DATA).ParentNodeIndex(0);
  DEF_GRAPH(partitioned_call1) {
                                 CHAIN(NODE("var", VARIABLE)->NODE("netoutput1", NETOUTPUT));
                                 CHAIN(NODE("const1", const1)->NODE("netoutput1", NETOUTPUT));
                                 CHAIN(NODE("data1", data1)->Ctrl()->NODE("refdata1", refdata1)->NODE("netoutput1", NETOUTPUT));
                               };

  const auto data2 = OP_CFG(DATA).ParentNodeIndex(1);
  const auto data3 = OP_CFG(DATA).ParentNodeIndex(2);
  const auto data4 = OP_CFG(DATA).ParentNodeIndex(3);
  const auto data5 = OP_CFG(DATA).ParentNodeIndex(4);
  const auto refnode = OP_CFG(ASSIGNADD).Attr(ATTR_NAME_REFERENCE, true).InNames({"ref"}).OutNames({"ref"});
  DEF_GRAPH(then_sub1) {
                         //CHAIN(NODE("data1", data1));
                         CHAIN(NODE("data2", data2)->NODE("refnode", refnode)->NODE("netoutput2", NETOUTPUT));
                         CHAIN(NODE("data3", data3)->NODE("netoutput2", NETOUTPUT));
                         CHAIN(NODE("data4", data4)->NODE("netoutput2", NETOUTPUT));
                         CHAIN(NODE("data5", data5)->NODE("netoutput2", NETOUTPUT));
                       };
  const auto data6 = OP_CFG(DATA).ParentNodeIndex(0);
  const auto data7 = OP_CFG(DATA).ParentNodeIndex(1);
  const auto data8 = OP_CFG(DATA).ParentNodeIndex(2);
  const auto data9 = OP_CFG(DATA).ParentNodeIndex(3);
  const auto data10 = OP_CFG(DATA).ParentNodeIndex(4);
  const auto refdata2 = OP_CFG(REFDATA).InCnt(1);
  const auto transdata1 = OP_CFG(TRANSDATA).Attr(REF_VAR_SRC_VAR_NAME, "refdata2");
  const auto transdata2 = OP_CFG(TRANSDATA).Attr(REF_VAR_SRC_VAR_NAME, "refdata2");// 若ref到var，要求var在根图
  DEF_GRAPH(else_sub1) {
                         CHAIN(NODE("data6", data6));
                         CHAIN(NODE("data7", data7));
                         CHAIN(NODE("data8", data8));
                         CHAIN(NODE("data9", data9));
                         CHAIN(NODE("data10", data10));

                         CHAIN(NODE("data7")->Ctrl()->NODE("refdata2", refdata2)->NODE("transdata1", transdata1)->NODE("netoutput3", NETOUTPUT));
                         CHAIN(NODE("op3", RELU)->NODE("transdata2", transdata2)->NODE("netoutput3", NETOUTPUT));
                         CHAIN(NODE("swt", LABELSWITCHBYINDEX)->NODE("netoutput3", NETOUTPUT));
                         CHAIN(NODE("op4", RELU)->NODE("netoutput3", NETOUTPUT));
                       };

  auto refdata_root = OP_CFG("RefData")
        .TensorDesc(FORMAT_ND, DT_FLOAT, {1,23,5,7})
        .InCnt(1)
        .OutCnt(1)
        .Attr(ATTR_NAME_INDEX, 0)
        .InNames({"x"})
        .OutNames({"y"})
        .Build("refdata_root");

  DEF_GRAPH(g1) {
                  CHAIN(NODE("op1", RELU)->NODE("if", IF, then_sub1, else_sub1));
                  CHAIN(NODE(refdata_root)->NODE("partitioned_call1", PARTITIONEDCALL, partitioned_call1)->NODE("if", IF, then_sub1, else_sub1)->NODE("op5", RELU));
                  CHAIN(NODE("partitioned_call1", PARTITIONEDCALL, partitioned_call1)->NODE("if", IF, then_sub1, else_sub1)->NODE("op5", RELU));
                  CHAIN(NODE("partitioned_call1", PARTITIONEDCALL, partitioned_call1)->NODE("if", IF, then_sub1, else_sub1)->NODE("op5", RELU));
                  CHAIN(NODE("op2", RELU)->NODE("if", IF, then_sub1, else_sub1)->NODE("op5", RELU));
                  CHAIN(NODE("op5", RELU)->NODE("netoutput", NETOUTPUT));
                };

  auto graph = ToComputeGraph(g1);
  AddParentIndexForSubGraphNetoutput(graph);
  // 如果不设置，会被设置为动态shape图
  AttrUtils::SetBool(graph, ATTR_NAME_NO_NEED_DYNAMIC_SHAPE_PARTITION, true);
  SetRefSrcVarName(graph, "transdata1", "refdata2"); // data pass will modify node name.
  SetRefSrcVarName(graph, "transdata2", "var");
  graph->TopologicalSorting();
  SetShapeForNodesOutputs(graph, {1, 1, 224, 224}, {"if"});
  SetSizeForAllNodes(graph);
  return graph;
}

/*
 *  unknow root graph               know subgraph
 *   ref_data1   partitioned_call1  +-----------------+
 *         \     / /                | var   constant1 |
 *         partitioned_call2        |   \    /        |
 *         |    |    |              |   netoutput1    |
 *         op1 op2 op3              +-----------------+
 *         |   |    |               know subgraph
 *         netoutput                +---------------------------------+
 *                                  | ref_data2  data2       data3    |
 *                                  |   |           |          |      |
 *                                  |transdata1 transdata2 transdata3 |
 *                                  |         \      |      /         |
 *                                  |            netoutput2           |
 *                                  +---------------------------------+
 */
ComputeGraphPtr MemConflictShareGraph::BuildRefDataGraph() {
  vector<int64_t> perm1(1 * 1 * 224 * 224, 0);
  GeTensorDesc tensor_desc1(GeShape(vector<int64_t>{1, 1, 224, 224}));
  GeTensorPtr const_tensor1 =
      std::make_shared<GeTensor>(tensor_desc1, reinterpret_cast<uint8_t *>(perm1.data()) , sizeof(int64_t)*perm1.size());
  auto const1 = OP_CFG(CONSTANTOP).Weight(const_tensor1);
  DEF_GRAPH(partitioned_call1) {
                                 CHAIN(NODE("var", VARIABLE)->NODE("netoutput1", NETOUTPUT));
                                 CHAIN(NODE("const1", const1)->NODE("netoutput1", NETOUTPUT));
                               };
  const auto ref_data2 = OP_CFG(REFDATA).InCnt(1).ParentNodeIndex(0).Attr(REF_VAR_SRC_VAR_NAME, "ref_data1");
  const auto transdata1 = OP_CFG(TRANSDATA).Attr(REF_VAR_SRC_VAR_NAME, "ref_data1");
  const auto transdata2 = OP_CFG(TRANSDATA).Attr(REF_VAR_SRC_VAR_NAME, "var");
  const auto data0 = OP_CFG(DATA).ParentNodeIndex(0);
  const auto data1 = OP_CFG(DATA).ParentNodeIndex(1);
  const auto data2 = OP_CFG(DATA).ParentNodeIndex(2);
  DEF_GRAPH(partitioned_call2) {
                                 CHAIN(NODE("data0", data0)->Ctrl()->NODE("ref_data2", ref_data2)->NODE("transdata1", transdata1)
                                           ->NODE("netoutput2", NETOUTPUT));
                                 CHAIN(NODE("data1", data1)->NODE("transdata2", transdata2)
                                           ->NODE("netoutput2", NETOUTPUT));
                                 CHAIN(NODE("data2", data2)->NODE("transdata3", TRANSDATA)
                                           ->NODE("netoutput2", NETOUTPUT));
                               };
  const auto ref_data1 = OP_CFG(REFDATA).InCnt(1);
  DEF_GRAPH(g1) {
                  CHAIN(NODE("ref_data1", ref_data1)->NODE("partitioned_call2", PARTITIONEDCALL, partitioned_call2)
                            ->NODE("op1", RELU)->NODE("netoutput", NETOUTPUT));
                  CHAIN(NODE("partitioned_call1", PARTITIONEDCALL, partitioned_call1)
                            ->NODE("partitioned_call2", PARTITIONEDCALL, partitioned_call2)
                            ->NODE("op2", RELU)->NODE("netoutput", NETOUTPUT));
                  CHAIN(NODE("partitioned_call1", PARTITIONEDCALL, partitioned_call1)
                            ->NODE("partitioned_call2", PARTITIONEDCALL, partitioned_call2)
                            ->NODE("op3", RELU)->NODE("netoutput", NETOUTPUT));
  };
  auto graph = ToComputeGraph(g1);
  AddParentIndexForSubGraphNetoutput(graph);
  // 设置为动态shape图
  AttrUtils::SetBool(graph, ATTR_NAME_NO_NEED_DYNAMIC_SHAPE_PARTITION, false);
  SetRefSrcVarName(graph, "ref_data2", "ref_data1");
  SetRefSrcVarName(graph, "transdata1", "ref_data1");
  SetRefSrcVarName(graph, "transdata2", "var");
  return graph;
}
/*
 *    const
 *      |
 *   phony_split (输出引用输入，且NoPadding连续输出内存)
 *     /  \
 *    a    b
 */
ComputeGraphPtr MemConflictShareGraph::BuildImmutableOutAndNoPaddingContinuousOutput() {
  const auto const_node = OP_CFG(CONSTANT).TensorDesc(FORMAT_NHWC, DT_FLOAT, {1, 1, 224, 448});
  const auto phony_split = OP_CFG("PhonySplit");
  DEF_GRAPH(g1) {
                  CHAIN(NODE("const_2", const_node)->NODE("netoutput", NETOUTPUT));
                  CHAIN(NODE("const", const_node)->NODE("phony_split", phony_split)->NODE("a", ADD)->NODE("netoutput", NETOUTPUT));
                  CHAIN(NODE("phony_split", phony_split)->NODE("a", ADD));
                };
  auto graph = ToComputeGraph(g1);
  auto ps = graph->FindNode("phony_split");
  ps->GetOpDescBarePtr()->MutableOutputDesc(0)->SetShape(GeShape(std::vector<int64_t>({1, 1, 112, 448})));
  ps->GetOpDescBarePtr()->MutableOutputDesc(1)->SetShape(GeShape(std::vector<int64_t>({1, 1, 112, 448})));

  // 如果不设置，会被设置为动态shape图
  AttrUtils::SetBool(graph, ATTR_NAME_NO_NEED_DYNAMIC_SHAPE_PARTITION, true);
  SetNoPaddingContinuousOutput(graph, "phony_split");
  return graph;
}

/*
 *    refdata
 *      |
 *    split (输出引用输入，且NoPadding连续输出内存)
 *     /  \
 *    a    b
 */
ComputeGraphPtr MemConflictShareGraph::BuildRefDataAndNoPaddingContinuousOutput() {
  const auto split = OP_CFG(SPLIT);
  const auto ref_data1 = OP_CFG(REFDATA).InCnt(1);
  DEF_GRAPH(g1) {
                  CHAIN(NODE("refdata2", ref_data1)->NODE("netoutput", NETOUTPUT));
                  CHAIN(NODE("refdata", ref_data1)->NODE("split", split)->NODE("a", ADD)->NODE("netoutput", NETOUTPUT));
                  CHAIN(NODE("split", split)->NODE("a", ADD));
                };
  auto graph = ToComputeGraph(g1);

  auto ref_data = graph->FindNode("refdata");
  ref_data->GetOpDescBarePtr()->MutableOutputDesc(0)->SetShape(GeShape(std::vector<int64_t>({1, 1, 224, 448})));
  auto ps = graph->FindNode("split");
  ps->GetOpDescBarePtr()->MutableOutputDesc(0)->SetShape(GeShape(std::vector<int64_t>({1, 1, 112, 448})));
  ps->GetOpDescBarePtr()->MutableOutputDesc(1)->SetShape(GeShape(std::vector<int64_t>({1, 1, 112, 448})));

  // 如果不设置，会被设置为动态shape图
  AttrUtils::SetBool(graph, ATTR_NAME_NO_NEED_DYNAMIC_SHAPE_PARTITION, true);
  SetNoPaddingContinuousOutput(graph, "split");
  return graph;
}

/*
 *     var
 *      +--ApplyMomentum-c
 *      |
 *   phony_split (输出引用输入，且NoPadding连续输出内存)
 *     /  \
 *    a    b
 */
ComputeGraphPtr MemConflictShareGraph::BuildVarAndNoPaddingContinuousOutputWithMultiReference() {
  const auto const_node = OP_CFG(CONSTANT).TensorDesc(FORMAT_NHWC, DT_FLOAT, {1, 1, 224, 448});
  const auto phony_split = OP_CFG("PhonySplit");
  const auto apply_momentum = OP_CFG(APPLYMOMENTUM).Attr(ATTR_NAME_REFERENCE, true).InNames({"ref"}).OutNames({"ref"});
  DEF_GRAPH(g1) {
                  CHAIN(NODE("const_2", const_node)->NODE("netoutput", NETOUTPUT));
                  CHAIN(NODE("var", VARIABLE)->NODE("phony_split", phony_split)->NODE("a", ADD)->NODE("netoutput", NETOUTPUT));
                  CHAIN(NODE("phony_split", phony_split)->NODE("a", ADD));
                  CHAIN(NODE("var", VARIABLE)->EDGE(0, 0)->NODE("apply_momentum", apply_momentum)->NODE("c", ADD)->NODE("netoutput", NETOUTPUT));
                };
  auto graph = ToComputeGraph(g1);
  // 如果不设置，会被设置为动态shape图
  AttrUtils::SetBool(graph, ATTR_NAME_NO_NEED_DYNAMIC_SHAPE_PARTITION, true);
  SetNoPaddingContinuousOutput(graph, "phony_split");
  return graph;
}

/*
 *   const  variable
 *      \     /
 *       concat
 */
ComputeGraphPtr MemConflictShareGraph::BuildImmutableOutAndContinuousInput() {
  DEF_GRAPH(g1) {
                  CHAIN(NODE("const", CONSTANT)->NODE("concat", CONCAT));
                  CHAIN(NODE("variable", VARIABLE)->NODE("concat", CONCAT));
                };
  auto graph = ToComputeGraph(g1);
  // 如果不设置，会被设置为动态shape图
  AttrUtils::SetBool(graph, ATTR_NAME_NO_NEED_DYNAMIC_SHAPE_PARTITION, true);
  SetContinuousInput(graph, "concat");
  return graph;
}
/*
 *   const  variable
 *      \     /
 *    phony_concat
 */
ComputeGraphPtr MemConflictShareGraph::BuildImmutableOutAndNoPaddingContinuousInput() {
  DEF_GRAPH(g1) {
                  CHAIN(NODE("const", CONSTANT)->NODE("phony_concat", CONCAT));
                  CHAIN(NODE("variable", VARIABLE)->NODE("phony_concat", CONCAT));
                };
  auto graph = ToComputeGraph(g1);
  // 如果不设置，会被设置为动态shape图
  AttrUtils::SetBool(graph, ATTR_NAME_NO_NEED_DYNAMIC_SHAPE_PARTITION, true);
  SetNoPaddingContinuousInput(graph, "phony_concat");
  return graph;
}
/*
 *  const
 *    |
 *   split  (continuous output, output reuse input)
 *   /  \
 *  a    b
 */
ComputeGraphPtr MemConflictShareGraph::BuildImmutableOutAndContinuousOutput(){
  DEF_GRAPH(g1) {
                  CHAIN(NODE("const", CONSTANT)->NODE("split", SPLIT)->NODE("a", RELU));
                  CHAIN(NODE("split", SPLIT)->NODE("b", RELU));
                };
  auto graph = ToComputeGraph(g1);
  // 如果不设置，会被设置为动态shape图
  AttrUtils::SetBool(graph, ATTR_NAME_NO_NEED_DYNAMIC_SHAPE_PARTITION, true);
  SetContinuousOutput(graph, "split");
  SetOutReuseInput(graph, "split");
  return graph;
}
/*
 *    var  const
 *     \    /
 *     assign
 *       |
 *   hcombroadcast
 *       / \
 *      a   b
 *      |   |
 *     netoutput
 */
ComputeGraphPtr MemConflictShareGraph::BuildImmutableOutAndContinuousByAssignOutput() {
  vector<int64_t> perm1(1 * 1 * 224 * 224, 0);
  GeTensorDesc tensor_desc1(GeShape(vector<int64_t>{1, 1, 224, 224}));
  GeTensorPtr const_tensor1 =
      std::make_shared<GeTensor>(tensor_desc1, reinterpret_cast<uint8_t *>(perm1.data()) , sizeof(int64_t)*perm1.size());
  auto constant = OP_CFG(CONSTANT).Weight(const_tensor1);
  auto hcombroadcast = OP_CFG(HCOMBROADCAST).Attr(ATTR_NAME_REFERENCE, true).InNames({"x"}).OutNames({"x", "y"});
  const auto assign = OP_CFG(ASSIGN).Attr(ATTR_NAME_REFERENCE, true).InNames({"ref"}).OutNames({"ref"});

  DEF_GRAPH(g1) {
                  CHAIN(NODE("var", VARIABLE)->NODE("assign", assign)->NODE("hcombroadcast", hcombroadcast)
                            ->NODE("a", RELU)->NODE("netoutput", NETOUTPUT));
                  CHAIN(NODE("const", constant)->NODE("assign", assign));
                  CHAIN(NODE("hcombroadcast", hcombroadcast)->NODE("b", RELU)->NODE("netoutput", NETOUTPUT));
                };
  auto graph = ToComputeGraph(g1);
  SetRefSrcVarName(graph, "assign", "var");
  SetContinuousOutput(graph, "hcombroadcast");

  // 如果不设置，会被设置为动态shape图
  AttrUtils::SetBool(graph, ATTR_NAME_NO_NEED_DYNAMIC_SHAPE_PARTITION, true);
  return graph;
}
/*
 *    var  const
 *     \    /
 *     assign   a
 *        \    /
 *          pc
 *           |
 *           b
 *           |
 *       netoutput
 */
ComputeGraphPtr MemConflictShareGraph::BuildImmutableOutAndNoPaddingContinuousInByAssignOutput() {
  vector<int64_t> perm1(1 * 1 * 224 * 224, 0);
  GeTensorDesc tensor_desc1(GeShape(vector<int64_t>{1, 1, 224, 224}));
  GeTensorPtr const_tensor1 =
      std::make_shared<GeTensor>(tensor_desc1, reinterpret_cast<uint8_t *>(perm1.data()) , sizeof(int64_t)*perm1.size());
  auto constant = OP_CFG(CONSTANT).Weight(const_tensor1);
  auto pc = OP_CFG(PHONYCONCAT);
  const auto assign = OP_CFG(ASSIGN).Attr(ATTR_NAME_REFERENCE, true).InNames({"ref"}).OutNames({"ref"});

  DEF_GRAPH(g1) {
                  CHAIN(NODE("var", VARIABLE)->NODE("assign", assign)->NODE("pc", pc)
                            ->NODE("b", RELU)->NODE("netoutput", NETOUTPUT));
                  CHAIN(NODE("const", constant)->NODE("assign", assign));
                  CHAIN(NODE("a", RELU)->NODE("pc", pc)->NODE("netoutput", NETOUTPUT));
                };
  auto graph = ToComputeGraph(g1);
  SetRefSrcVarName(graph, "assign", "var");
  SetNoPaddingContinuousInput(graph, "pc");

  // 如果不设置，会被设置为动态shape图
  AttrUtils::SetBool(graph, ATTR_NAME_NO_NEED_DYNAMIC_SHAPE_PARTITION, true);
  return graph;
}
/*
 * const var1    var2
 *   \   /        |
 *    hcom     phonysplit
 *    / \        /  \
 *   a   b      c    d
 *   |     \  /      |
 *   +---netoutput---+
 */
ComputeGraphPtr MemConflictShareGraph::BuildImmutableOutAndNoPaddingAndContinuousByRefOutput() {
  vector<int64_t> perm1(1 * 1 * 224 * 224, 0);
  GeTensorDesc tensor_desc1(GeShape(vector<int64_t>{1, 1, 224, 224}));
  GeTensorPtr const_tensor1 =
      std::make_shared<GeTensor>(tensor_desc1, reinterpret_cast<uint8_t *>(perm1.data()) , sizeof(int64_t)*perm1.size());
  auto constant1 = OP_CFG(CONSTANT).Weight(const_tensor1);

  DEF_GRAPH(g1) {
                  CHAIN(NODE("const", constant1)->NODE("hcom", HCOMALLGATHER)->NODE("a", RELU)->NODE("netoutput", NETOUTPUT));
                  CHAIN(NODE("hcom", HCOMALLGATHER)->NODE("b", RELU)->NODE("netoutput", NETOUTPUT));
                  CHAIN(NODE("var1", VARIABLE)->NODE("hcom", HCOMALLGATHER));

                  CHAIN(NODE("var2", VARIABLE)->NODE("phonysplit", PHONYSPLIT)->NODE("c", RELU)->NODE("netoutput", NETOUTPUT));
                  CHAIN(NODE("phonysplit", PHONYSPLIT)->NODE("d", RELU)->NODE("netoutput", NETOUTPUT));
                };
  auto graph = ToComputeGraph(g1);
  auto var2 = graph->FindNode("var2");
  var2->GetOpDescBarePtr()->MutableOutputDesc(0)->SetShape(GeShape(std::vector<int64_t>({1, 1, 224, 448})));
  auto var1 = graph->FindNode("var1");
  var1->GetOpDescBarePtr()->MutableOutputDesc(0)->SetShape(GeShape(std::vector<int64_t>({1, 1, 224, 448})));

  auto ps = graph->FindNode("phonysplit");
  ps->GetOpDescBarePtr()->MutableOutputDesc(0)->SetShape(GeShape(std::vector<int64_t>({1, 1, 112, 448})));
  ps->GetOpDescBarePtr()->MutableOutputDesc(1)->SetShape(GeShape(std::vector<int64_t>({1, 1, 112, 448})));
  // 如果不设置，会被设置为动态shape图
  AttrUtils::SetBool(graph, ATTR_NAME_NO_NEED_DYNAMIC_SHAPE_PARTITION, true);
  SetContinuousOutput(graph, "hcom");
  SetOutReuseInput(graph, "hcom");
  auto hcom_node = graph->FindNode("hcom");
  // 为了输出能和输入同符号
  auto output_tensor = hcom_node->GetOpDesc()->MutableOutputDesc(1);
  if (output_tensor != nullptr) {
    TensorUtils::SetReuseInput(*output_tensor, true);
    TensorUtils::SetReuseInputIndex(*output_tensor, 1U);
  }
  hcom_node->GetOpDescBarePtr()->MutableOutputDesc(0)->SetShape(GeShape(std::vector<int64_t>({1, 1, 224, 448})));
  hcom_node->GetOpDescBarePtr()->MutableOutputDesc(1)->SetShape(GeShape(std::vector<int64_t>({1, 1, 224, 448})));
  SetNoPaddingContinuousOutput(graph, "phonysplit");
  return graph;
}

/*
 *      add
 *       /\
 *     netoutput
 */
ComputeGraphPtr MemConflictShareGraph::BuildUserOutAndUserOutGraph() {
  DEF_GRAPH(g1) {
                  CHAIN(NODE("add", ADD)->NODE("netoutput", NETOUTPUT));
                  CHAIN(NODE("add", ADD)->EDGE(0, 1)->NODE("netoutput", NETOUTPUT));
                };
  return ToComputeGraph(g1);
}

/*
 *      add
 *       /\
 *   partitioned_call1  +-------------+
 *      |  |            | data1  data2|
 *     netoutput        |  |      |   |
 *                      |  netoutput1 |
 *                      +-------------+
 */
ComputeGraphPtr MemConflictShareGraph::BuildUserOutAndUserOutGraphWithSubGraph() {
  const auto data1 = OP_CFG(DATA).ParentNodeIndex(0);
  const auto data2 = OP_CFG(DATA).ParentNodeIndex(1);
  DEF_GRAPH(partitioned_call1) {
                                 CHAIN(NODE("data1", data1)->NODE("netoutput1", NETOUTPUT));
                                 CHAIN(NODE("data2", data2)->NODE("netoutput1", NETOUTPUT));
                               };
  DEF_GRAPH(g1) {
                  CHAIN(NODE("add", ADD)->NODE("partitioned_call1", PARTITIONEDCALL, partitioned_call1));
                  CHAIN(NODE("add", ADD)->EDGE(0, 1)->NODE("partitioned_call1", PARTITIONEDCALL, partitioned_call1));
                  CHAIN(NODE("partitioned_call1", PARTITIONEDCALL, partitioned_call1)->NODE("netoutput", NETOUTPUT));
                  CHAIN(NODE("partitioned_call1", PARTITIONEDCALL, partitioned_call1)->NODE("netoutput", NETOUTPUT));
                };
  auto graph = ToComputeGraph(g1);
  AddParentIndexForSubGraphNetoutput(graph);
  return graph;
}

/*
 * unknown graph
 *      op1
 *       |               known sub graph
 * partitioned_call      +-------------+
 *       |               |data  op3    |
 *      op2              |      / \    |
 *                       |    netoutput|
 *                       +-------------+
 */
ComputeGraphPtr MemConflictShareGraph::BuildUserOutAndUserOutGraphInSubGraph() {
  const auto sub_data = OP_CFG(DATA).ParentNodeIndex(0);
  DEF_GRAPH(sub_1) {
                     CHAIN(NODE("data", sub_data));
                     CHAIN(NODE("op3", ADD)->NODE("netoutput", NETOUTPUT));
                     CHAIN(NODE("op3", ADD)->EDGE(0, 1)->NODE("netoutput", NETOUTPUT));
                   };
  DEF_GRAPH(g1) {
                  CHAIN(NODE("op1", ADD)->NODE("partitioned_call", PARTITIONEDCALL, sub_1)->NODE("op2", ADD));
                };
  auto graph = ToComputeGraph(g1);
  graph->SetGraphUnknownFlag(true);
  return graph;
}

/*
 *      const
 *       |
 *     netoutput
 */
ComputeGraphPtr MemConflictShareGraph::BuildUserOutAndConstOutGraph() {
  DEF_GRAPH(g1) {
                  CHAIN(NODE("const", CONSTANT)->NODE("netoutput", NETOUTPUT));
                };
  return ToComputeGraph(g1);
}
/*
 *      variable
 *       |
 *     netoutput
 */
ComputeGraphPtr MemConflictShareGraph::BuildUserOutAndVariableGraph() {
  DEF_GRAPH(g1) {
                  CHAIN(NODE("variable", VARIABLE)->NODE("netoutput", NETOUTPUT));
                };
  return ToComputeGraph(g1);
}

/*
 *      refdata
 *       |
 *     netoutput
 */
ComputeGraphPtr MemConflictShareGraph::BuildUserOutAndRefdataGraph() {
  DEF_GRAPH(g1) {
                  CHAIN(NODE("refdata", REFDATA)->NODE("netoutput", NETOUTPUT));
                };
  return ToComputeGraph(g1);
}
/*
 *   partitioned_call1
 *         |           +-----------+
 *         |           |  const1   |
 *         |           |   |       |
 *         |           | netoutput1|
 *         |           +-----------+
 *       netoutput
 */
ComputeGraphPtr MemConflictShareGraph::BuildUserOutAndConstOutGraphWithSubGraph() {

  DEF_GRAPH(partitioned_call1) {
                                 CHAIN(NODE("constant1", CONSTANTOP)->NODE("netoutput1", NETOUTPUT));
                               };
  DEF_GRAPH(g1) {
                  CHAIN(NODE("partitioned_call1", PARTITIONEDCALL, partitioned_call1)->NODE("netoutput", NETOUTPUT));
                };
  auto graph = ToComputeGraph(g1);
  AddParentIndexForSubGraphNetoutput(graph);
  return graph;
}

/*
 * unknown graph
 *      op1
 *       |               known sub graph
 * partitioned_call      +-------------+
 *       |               |data  constant|
 *      op2              |       |     |
 *                       |    netoutput|
 *                       +-------------+
 */
ComputeGraphPtr MemConflictShareGraph::BuildUserOutAndConstOutGraphInSubgraph() {
  const auto sub_data = OP_CFG(DATA).ParentNodeIndex(0);
  DEF_GRAPH(sub_1) {
                     CHAIN(NODE("data", sub_data));
                     CHAIN(NODE("constant", CONSTANTOP)->NODE("netoutput", NETOUTPUT));
                   };
  DEF_GRAPH(g1) {
                  CHAIN(NODE("op1", ADD)->NODE("partitioned_call", PARTITIONEDCALL, sub_1)->NODE("op2", ADD));
                };
  auto graph = ToComputeGraph(g1);
  graph->SetGraphUnknownFlag(true);
  return graph;
}

/*
 *
 * unknown graph
 *      op1
 *       |               known sub graph
 * partitioned_call      +-----------+
 *       |               |   data    |
 *      op2              |    |      |
 *                       |    op3    |
 *                       |    |      |
 *                       |  switch   |
 *                       |    |      |
 *                       | netoutput |
 *                       +-----------+
 */
ComputeGraphPtr MemConflictShareGraph::BuildUserOutAndNotSupportRefreshOutWithSubgraphDataGraph() {
  const auto sub_data = OP_CFG(DATA).ParentNodeIndex(0);
  DEF_GRAPH(sub_1) {
                     CHAIN(NODE("data", sub_data)->NODE("op3", ADD)->NODE("switch", STREAMSWITCH)
                               ->NODE("netoutput", NETOUTPUT));
                   };
  DEF_GRAPH(g1) {
                  CHAIN(NODE("op1", ADD)->NODE("partitioned_call", PARTITIONEDCALL, sub_1)->NODE("op2", ADD));
                };
  auto graph = ToComputeGraph(g1);
  graph->SetGraphUnknownFlag(true);
  return graph;
}

/*
 *    op1
 *     +--switch (not support refresh address)
 *     |
 *   netoutput
 */
ComputeGraphPtr MemConflictShareGraph::BuildUserOutAndNotSupportRefreshInGraph() {
  DEF_GRAPH(g1) {
                  CHAIN(NODE("op1", ADD)->NODE("switch", STREAMSWITCH));
                  CHAIN(NODE("op1", ADD)->EDGE(0, 0)->NODE("netoutput", NETOUTPUT));
                };
  auto graph = ToComputeGraph(g1);
  return graph;
}
/*
 *    a
 *    |
 * hcombroadcast 输入不支持地址刷新，输出不支持地址刷新，且输出引用输入
 *    |
 *  Netoutput
 */
ComputeGraphPtr MemConflictShareGraph::BuildUserOutAndNotSupportRefreshInByRefGraph() {
  auto hcombroadcast = OP_CFG(HCOMBROADCAST).Attr(ATTR_NAME_REFERENCE, true).InNames({"x"}).OutNames({"x"});
  DEF_GRAPH(g1) {
                  CHAIN(NODE("a", ADD)->NODE("hcombroadcast", hcombroadcast)->NODE("netoutput", NETOUTPUT));
                };
  auto graph = ToComputeGraph(g1);
  SetEngine(graph, {"hcombroadcast"}, kEngineNameHccl);
  graph->SetGraphUnknownFlag(false);
  return graph;
}
/*
 *          a
 *         / \
 * netoutput swt
 *             \
 *              b
 */
ComputeGraphPtr MemConflictShareGraph::BuildUserOutAndNotSupportRefreshOutOneOutMultiRefGraph() {
  DEF_GRAPH(g1) {
                  CHAIN(NODE("a", ADD)->NODE("swt", STREAMSWITCH)->NODE("b", ADD));
                  CHAIN(NODE("a", ADD)->EDGE(0, 0)->NODE("netoutput", NETOUTPUT));
                };
  auto graph = ToComputeGraph(g1);
  graph->SetGraphUnknownFlag(false);
  return graph;
}

/*
 *          a
 *         / \
 * netoutput hcom
 *             \
 *              b
 */
ComputeGraphPtr MemConflictShareGraph::BuildUserOutAndNotSupportRefreshOutByRefGraph() {
  DEF_GRAPH(g1) {
                  CHAIN(NODE("a", ADD)->NODE("hcom", HCOMBROADCAST)->NODE("b", ADD));
                  CHAIN(NODE("a", ADD)->EDGE(0, 0)->NODE("netoutput", NETOUTPUT));
                };
  auto graph = ToComputeGraph(g1);
  SetEngine(graph, {"hcom"}, kEngineNameHccl);
  SetOutReuseInput(graph, "hcom");
  graph->SetGraphUnknownFlag(false);
  return graph;
}

/*
 *    op1
 *     +--hcombroadcast (p2p input)
 *     |   |
 *   netoutput
 *
 *   ||
 *   \/
 *
 *   op1
 *     +--identity--hcombroadcast (p2p input)
 *     |            /
 *     |    identity
 *     |    /
 *   netoutput
 *
 * 用例场景：rts特殊类型内存与用户输出相接，hcombroadcast是输出引用输入，需要在netoutput前插入identity。
 * 步骤：
 * step 1. 按照用例场景构图
 * 期望：构图成功
 * step 2. 执行图编译
 * 期望： 图编译返回成功
 * step 3. 校验
 * 期望：netoutput前插入identity
 */
ComputeGraphPtr MemConflictShareGraph::BuildUserOutAndRtsSpecialInGraph() {
  auto hcombroadcast = OP_CFG(HCOMBROADCAST).Attr(ATTR_NAME_REFERENCE, true).InNames({"x"}).OutNames({"x"});
  DEF_GRAPH(g1) {
                  CHAIN(NODE("data", DATA)->NODE("op1", RELU)->NODE("hcombroadcast", hcombroadcast)->EDGE(0, 1)->NODE("netoutput", NETOUTPUT));
                  CHAIN(NODE("op1", RELU)->EDGE(0, 0)->NODE("netoutput", NETOUTPUT));
                };
  std::vector<int64_t> in_mem_type {RT_MEMORY_P2P_DDR};
  auto graph = ToComputeGraph(g1);
  auto op1 = graph->FindNode("hcombroadcast");
  (void)ge::AttrUtils::SetListInt(op1->GetOpDescBarePtr(), ATTR_NAME_INPUT_MEM_TYPE_LIST, in_mem_type);
  (void)ge::AttrUtils::SetListInt(op1->GetOpDescBarePtr(), ATTR_NAME_OUTPUT_MEM_TYPE_LIST, in_mem_type);
  // 如果不设置，会被设置为动态shape图
  AttrUtils::SetBool(graph, ATTR_NAME_NO_NEED_DYNAMIC_SHAPE_PARTITION, true);
  return graph;
}

/*
 *   hcomallreduce (p2p output)
 *     |
 *   netoutput
 */
ComputeGraphPtr MemConflictShareGraph::BuildUserOutAndRtsSpecialOutGraph() {
  DEF_GRAPH(g1) {
                  CHAIN(NODE("hcomallreduce", HCOMALLREDUCE)->EDGE(0, 0)->NODE("netoutput", NETOUTPUT));
                };
  std::vector<int64_t> mem_type {RT_MEMORY_P2P_DDR};
  auto graph = ToComputeGraph(g1);
  auto op1 = graph->FindNode("hcomallreduce");
  (void)ge::AttrUtils::SetListInt(op1->GetOpDescBarePtr(), ATTR_NAME_OUTPUT_MEM_TYPE_LIST, mem_type);
  return graph;
}

/*
 *      op1                      partitioned_call2    op3         op4
 *       │                          ┌───────────┐      │           │
 *     ┌─┼───────┐                  │ constant1 │      ├────┐      ├────┐
 *     │ │       │                  │     │     │      │    │      │    │
 *     │ │  partitioned_call1       │     │     │      │   swt     │   hcom
 *     │ │  ┌────────────┐          │ netoutput2│      │    │      │    │
 *     │ │  │   data1    │          └─────┬─────┘      │    │      │    │
 *     │ │  │    ┼       │                │            │    │      │    │
 *     │ │  │  netoutput1│                │            │    │      │    │
 *     │ │  └────┬───────┘                │            │    │      │    │
 *     │ │       │                        │            │    │      │    │
 *     │ │       └───────────────┐ ┌──────┘            │    │      │    │
 *     │ │                       │ │                   │    │      │    │
 *     │ └─────────────────────┐ │ │ ┌─────────────────┘    │      │    │
 *     │                       │ │ │ │                      │      │    │
 *     └─────────────────────┐ │ │ │ │  ┌───────────────────┘      │    │
 *                           │ │ │ │ │  │                          │    │
 *                           │ │ │ │ │  │  ┌───────────────────────┘    │
 *                           │ │ │ │ │  │  │                            │
 *                           │ │ │ │ │  │  │  ┌─────────────────────────┘
 *                          0│1│2│3│4│ 5│ 6│ 7│
 *                              netoutput
 */
ComputeGraphPtr MemConflictShareGraph::BuildUserOutAndOtherConflictTypeGraph() {
  const auto data1 = OP_CFG(DATA).ParentNodeIndex(0);
  DEF_GRAPH(partitioned_call1) {
                                 CHAIN(NODE("data1", data1)->NODE("netoutput1", NETOUTPUT));
                               };

  vector<int64_t> perm1(1 * 1 * 224 * 224, 0);
  GeTensorDesc tensor_desc1(GeShape(vector<int64_t>{1, 1, 224, 224}));
  GeTensorPtr const_tensor1 =
      std::make_shared<GeTensor>(tensor_desc1, reinterpret_cast<uint8_t *>(perm1.data()) , sizeof(int64_t)*perm1.size());
  auto constant1 = OP_CFG(CONSTANTOP).Weight(const_tensor1);

  DEF_GRAPH(partitioned_call2) {
                                 CHAIN(NODE("constant1", constant1)->NODE("netoutput2", NETOUTPUT));
                               };
  DEF_GRAPH(g1) {
                  CHAIN(NODE("op1", RELU)->EDGE(0, 0)->NODE("netoutput", NETOUTPUT));
                  CHAIN(NODE("op1", RELU)->EDGE(0, 1)->NODE("netoutput", NETOUTPUT));
                  CHAIN(NODE("op1", RELU)->EDGE(0, 0)->NODE("partitioned_call1", PARTITIONEDCALL, partitioned_call1)
                        ->EDGE(0, 2)->NODE("netoutput", NETOUTPUT));
                  CHAIN(NODE("partitioned_call2", PARTITIONEDCALL, partitioned_call2)
                            ->EDGE(0, 3)->NODE("netoutput", NETOUTPUT));
                  CHAIN(NODE("op3", RELU)->EDGE(0, 4)->NODE("netoutput", NETOUTPUT));
                  CHAIN(NODE("op3", RELU)->EDGE(0, 0)->NODE("swt", LABELSWITCHBYINDEX)
                            ->EDGE(0, 5)->NODE("netoutput", NETOUTPUT));
                  CHAIN(NODE("op4", RELU)->EDGE(0, 6)->NODE("netoutput", NETOUTPUT));
                  CHAIN(NODE("op4", RELU)->EDGE(0, 0)->NODE("hcom", HCOMALLREDUCE)
                            ->EDGE(0, 7)->NODE("netoutput", NETOUTPUT));
  };
  auto graph = ToComputeGraph(g1);
  AddParentIndexForSubGraphNetoutput(graph);

  auto hcom = graph->FindNode("hcom");
  hcom->GetOpDesc()->SetOpKernelLibName("ops_kernel_info_hccl");
  AttrUtils::SetBool(hcom->GetOpDesc(), ATTR_NAME_MODIFY_INPUT, true);

  SetRtsSpecialTypeInput(graph, "hcom", RT_MEMORY_P2P_DDR);
  SetRtsSpecialTypeOutput(graph, "hcom", RT_MEMORY_P2P_DDR);

  // 如果不设置，会被设置为动态shape图
  AttrUtils::SetBool(graph, ATTR_NAME_NO_NEED_DYNAMIC_SHAPE_PARTITION, true);
  return graph;
}

/*
 *   a             b
 *   |             |
 *   +-------+     |
 *   |        \    /
 *   |      hcomallreduce
 * netoutput
 */
ComputeGraphPtr MemConflictShareGraph::BuildUserOutAndContinuousInputGraph() {
  DEF_GRAPH(g1) {
                  CHAIN(NODE("a", ADD)->EDGE(0, 0)->NODE("hcomallreduce", HCOMALLREDUCE));
                  CHAIN(NODE("a", ADD)->EDGE(0, 0)->NODE("netoutput", NETOUTPUT));
                  CHAIN(NODE("b", ADD)->EDGE(0, 1)->NODE("hcomallreduce", HCOMALLREDUCE));
                };
  auto graph = ToComputeGraph(g1);
  SetContinuousInput(graph, "hcomallreduce");
  // 如果不设置，会被设置为动态shape图
  AttrUtils::SetBool(graph, ATTR_NAME_NO_NEED_DYNAMIC_SHAPE_PARTITION, true);
  return graph;
}
/*
 *   a             b
 *   |             |
 *   +-------+     |
 *   |        \    /
 *   |      phonyconcat
 * netoutput
 */
ComputeGraphPtr MemConflictShareGraph::BuildUserOutAndNoPaddingContinuousInputGraph() {
  DEF_GRAPH(g1) {
                  CHAIN(NODE("a", ADD)->EDGE(0, 0)->NODE("phonyconcat", PHONYCONCAT));
                  CHAIN(NODE("a", ADD)->EDGE(0, 0)->NODE("netoutput", NETOUTPUT));
                  CHAIN(NODE("b", ADD)->EDGE(0, 1)->NODE("phonyconcat", PHONYCONCAT));
                };
  auto graph = ToComputeGraph(g1);
  SetNoPaddingContinuousInput(graph, "phonyconcat");
  // 如果不设置，会被设置为动态shape图
  AttrUtils::SetBool(graph, ATTR_NAME_NO_NEED_DYNAMIC_SHAPE_PARTITION, true);
  return graph;
}
/*
 * data1     data2 data3          data4
 *   |         |     |             |
 *   a         b     c             d
 *   |         |     |             |
 *   +---+     |     +-------+     |
 *   |    \    /     |        \    /
 *   |     hcom      |      phonyconcat
 *   |      |        |          |
 *   |      +--+  +--+          |
 *   +-------+ |  | +-----------+
 *           | |  | |
 *           netoutput
 */
ComputeGraphPtr MemConflictShareGraph::BuildUserOutAndNoPaddingAndContinuousInputGraph() {
  DEF_GRAPH(g1) {
                  CHAIN(NODE("data1", DATA)->NODE("a", RELU)->NODE("hcom", HCOMALLGATHER)->EDGE(0, 1)->NODE("netoutput", NETOUTPUT));
                  CHAIN(NODE("a", RELU)->EDGE(0, 0)->NODE("netoutput", NETOUTPUT));
                  CHAIN(NODE("data2", DATA)->NODE("b", RELU)->EDGE(0, 1)->NODE("hcom", HCOMALLGATHER));

                  CHAIN(NODE("data3", DATA)->NODE("c", RELU)->NODE("phonyconcat", PHONYCONCAT)->EDGE(0, 3)->NODE("netoutput", NETOUTPUT));
                  CHAIN(NODE("c", RELU)->EDGE(0, 2)->NODE("netoutput", NETOUTPUT));
                  CHAIN(NODE("data4", DATA)->NODE("d", RELU)->EDGE(0, 1)->NODE("phonyconcat", PHONYCONCAT));
                };
  auto graph = ToComputeGraph(g1);
  auto pc = graph->FindNode("phonyconcat");
  pc->GetOpDescBarePtr()->MutableOutputDesc(0)->SetShape(GeShape(std::vector<int64_t>({2, 1, 224, 224})));
  auto c = graph->FindNode("c");
  c->GetOpDescBarePtr()->MutableOutputDesc(0)->SetShape(GeShape(std::vector<int64_t>({1, 1, 224, 224})));
  auto d = graph->FindNode("d");
  d->GetOpDescBarePtr()->MutableOutputDesc(0)->SetShape(GeShape(std::vector<int64_t>({1, 1, 224, 224})));

  SetContinuousInput(graph, "hcom");
  SetNoPaddingContinuousInput(graph, "phonyconcat");
  // 如果不设置，会被设置为动态shape图
  AttrUtils::SetBool(graph, ATTR_NAME_NO_NEED_DYNAMIC_SHAPE_PARTITION, true);
  return graph;
}

/*
 *    a
 *    |
 *   refnode   b
 *        \  /
 *         pc
 */
ComputeGraphPtr MemConflictShareGraph::BuildConnectToNoPaddingContinuousInputThroughtRefNodeGraph() {
  const auto refnode = OP_CFG(ASSIGNADD).Attr(ATTR_NAME_REFERENCE, true).InNames({"ref"}).OutNames({"ref"});
  DEF_GRAPH(g1) {
                  CHAIN(NODE("a", RELU)->NODE("refnode", refnode)->NODE("pc", PHONYCONCAT)->NODE("netoutput", NETOUTPUT));
                  CHAIN(NODE("b", RELU)->EDGE(0, 1)->NODE("pc", PHONYCONCAT));
                };
  auto graph = ToComputeGraph(g1);
  SetNoPaddingContinuousInput(graph, "pc");
  // 如果不设置，会被设置为动态shape图
  AttrUtils::SetBool(graph, ATTR_NAME_NO_NEED_DYNAMIC_SHAPE_PARTITION, true);
  return graph;
}

/*
 *    a
 *    |
 *   refnode   b
 *        \  /
 *         hcom
 */
ComputeGraphPtr MemConflictShareGraph::BuildConnectToContinuousInputThroughtRefNodeGraph() {
  const auto refnode = OP_CFG(ASSIGNADD).Attr(ATTR_NAME_REFERENCE, true).InNames({"ref"}).OutNames({"ref"});
  DEF_GRAPH(g1) {
                  CHAIN(NODE("a", RELU)->NODE("refnode", refnode)->NODE("hcom", HCOMALLREDUCE)->NODE("netoutput", NETOUTPUT));
                  CHAIN(NODE("b", RELU)->EDGE(0, 1)->NODE("hcom", HCOMALLREDUCE));
                };
  auto graph = ToComputeGraph(g1);
  SetContinuousInput(graph, "hcom");
  // 如果不设置，会被设置为动态shape图
  AttrUtils::SetBool(graph, ATTR_NAME_NO_NEED_DYNAMIC_SHAPE_PARTITION, true);
  return graph;
}

/*
 *         hcom
 *        /   \
 *   refnode   a
 *        \
 *         b
 */
ComputeGraphPtr MemConflictShareGraph::BuildContinousOutConnectRefNodeGraph() {
  const auto refnode = OP_CFG(ASSIGNADD).Attr(ATTR_NAME_REFERENCE, true).InNames({"ref"}).OutNames({"ref"});
  DEF_GRAPH(g1) {
                  CHAIN(NODE("hcom", HCOMALLREDUCE)->NODE("refnode", refnode)->NODE("b", RELU)->NODE("netoutput", NETOUTPUT));
                  CHAIN(NODE("hcom", HCOMALLREDUCE)->NODE("a", RELU));
                };
  auto graph = ToComputeGraph(g1);
  SetContinuousOutput(graph, "hcom");
  // 如果不设置，会被设置为动态shape图
  AttrUtils::SetBool(graph, ATTR_NAME_NO_NEED_DYNAMIC_SHAPE_PARTITION, true);
  return graph;
}

/*
 *         split
 *        /   \
 *   refnode   a
 *        \
 *         b
 */
ComputeGraphPtr MemConflictShareGraph::BuildNopaddingContinousOutConnectRefNodeGraph() {
  const auto refnode = OP_CFG(ASSIGNADD).Attr(ATTR_NAME_REFERENCE, true).InNames({"ref"}).OutNames({"ref"});
  DEF_GRAPH(g1) {
                  CHAIN(NODE("split", PHONYSPLIT)->NODE("refnode", refnode)->NODE("b", RELU)->NODE("netoutput", NETOUTPUT));
                  CHAIN(NODE("split", PHONYSPLIT)->NODE("a", RELU));
                };
  auto graph = ToComputeGraph(g1);
  SetNoPaddingContinuousOutput(graph, "split");
  // 如果不设置，会被设置为动态shape图
  AttrUtils::SetBool(graph, ATTR_NAME_NO_NEED_DYNAMIC_SHAPE_PARTITION, true);
  return graph;
}

/*
 *      data
 *       |
 *      split
 *      /    \
 * netoutput  b
 */
ComputeGraphPtr MemConflictShareGraph::BuildUserOutAndContinuousOutputGraph() {
  DEF_GRAPH(g1) {
                  CHAIN(NODE("data", DATA)->NODE("split", SPLIT)->NODE("netoutput", NETOUTPUT));
                  CHAIN(NODE("split", SPLIT)->NODE("b", RELU)->NODE("netoutput", NETOUTPUT));
                };
  auto graph = ToComputeGraph(g1);
  SetContinuousOutput(graph, "split");
  // 如果不设置，会被设置为动态shape图
  AttrUtils::SetBool(graph, ATTR_NAME_NO_NEED_DYNAMIC_SHAPE_PARTITION, true);
  return graph;
}
/*
 *      hcom
 *       ||...
 *      netoutput
 */
ComputeGraphPtr MemConflictShareGraph::BuildUserOutAndALotOfContinuousOutputGraph() {
  auto hcom = OP_CFG(HCOMALLREDUCE)
                    .OutCnt(2000)
                    .Build("HcomReduceScatter_55_abcdefghijklmnopqrstuvwxwzdddddddddddddddddddddddddddddddddddddddddddaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaafffffffffffffffffff");
  auto netoutput = OP_CFG(NETOUTPUT)
                    .InCnt(2000)
                    .Build("HcomReduceScatter_54_aaaaaaaddddddddjjjjjjjjdddddddddddllllllzjjjjjjjjjjjjjjeeeeeeeeezzzzzzzzzzzzzzzzzzzzzjjjjjjjjjjjjiiiiiiiallllllldd");
  DEF_GRAPH(g1) {
                  CHAIN(NODE(hcom)->NODE(netoutput));
                };
  auto graph = ToComputeGraph(g1);
  SetContinuousOutput(graph, "hcom");
  auto hcom_node = graph->FindFirstNodeMatchType(HCOMALLREDUCE);
  auto netoutput_node = graph->FindFirstNodeMatchType(NETOUTPUT);

  for (size_t i = 1U; i < hcom_node->GetOpDesc()->GetOutputsSize(); ++i) {
    GraphUtils::AddEdge(hcom_node->GetOutDataAnchor(i), netoutput_node->GetInDataAnchor(i));
  }
  // 如果不设置，会被设置为动态shape图
  AttrUtils::SetBool(graph, ATTR_NAME_NO_NEED_DYNAMIC_SHAPE_PARTITION, true);
  return graph;
}
/*
 *       a
 *       |
 *  phony_split
 *      /    \
 * netoutput  b
 */
ComputeGraphPtr MemConflictShareGraph::BuildUserOutAndNoPaddingContinuousOutputGraph() {
  DEF_GRAPH(g1) {
                  CHAIN(NODE("a", RELU)->NODE("phony_split", SPLIT)->NODE("netoutput", NETOUTPUT));
                  CHAIN(NODE("phony_split", SPLIT)->NODE("b", RELU)->NODE("netoutput", NETOUTPUT));
                };
  auto graph = ToComputeGraph(g1);
  SetNoPaddingContinuousOutput(graph, "phony_split");
  // 如果不设置，会被设置为动态shape图
  AttrUtils::SetBool(graph, ATTR_NAME_NO_NEED_DYNAMIC_SHAPE_PARTITION, true);
  return graph;
}
/*
 * data1  data2
 *   |      |
 *   a      b
 *   \      /
 *   phony_concat (NoPaddingContinuousInput,
 *      |           and otuput ref input)
 *    netoutput
 */
ComputeGraphPtr MemConflictShareGraph::BuildUserOutAndNoPaddingContinuousInputByReferenceGraph() {
  DEF_GRAPH(g1) {
                  CHAIN(NODE("data1", DATA)->NODE("a", RELU)->NODE("phony_concat", PHONYCONCAT)->NODE("netoutput", NETOUTPUT));
                  CHAIN(NODE("data2", DATA)->NODE("b", RELU)->NODE("phony_concat", PHONYCONCAT));
                };
  auto graph = ToComputeGraph(g1);
  auto a = graph->FindNode("a");
  a->GetOpDescBarePtr()->MutableOutputDesc(0)->SetShape(GeShape(std::vector<int64_t>({1, 1, 112, 448})));
  auto b = graph->FindNode("b");
  b->GetOpDescBarePtr()->MutableOutputDesc(0)->SetShape(GeShape(std::vector<int64_t>({1, 1, 112, 448})));

  auto pc = graph->FindNode("phony_concat");
  pc->GetOpDescBarePtr()->MutableOutputDesc(0)->SetShape(GeShape(std::vector<int64_t>({1, 1, 224, 448})));

  SetNoPaddingContinuousInput(graph, "phony_concat");
  // 如果不设置，会被设置为动态shape图
  AttrUtils::SetBool(graph, ATTR_NAME_NO_NEED_DYNAMIC_SHAPE_PARTITION, true);
  return graph;
}
/*
 * data1  data2
 *   |      |
 *   a      b
 *   \      /
 *   concat (ContinuousInput,
 *      |    and otuput ref input)
 *    netoutput
 */
ComputeGraphPtr MemConflictShareGraph::BuildUserOutAndContinuousInputByReferenceGraph() {
  DEF_GRAPH(g1) {
                  CHAIN(NODE("data1", DATA)->NODE("a", RELU)->NODE("concat", CONCAT)->NODE("netoutput", NETOUTPUT));
                  CHAIN(NODE("data2", DATA)->NODE("b", RELU)->NODE("concat", CONCAT));
                };
  auto graph = ToComputeGraph(g1);
  SetContinuousInput(graph, "concat");
  SetOutReuseInput(graph, "concat");
  // 如果不设置，会被设置为动态shape图
  AttrUtils::SetBool(graph, ATTR_NAME_NO_NEED_DYNAMIC_SHAPE_PARTITION, true);
  return graph;
}
/*
 *   const
 *    |
 *   swt
 */
ComputeGraphPtr MemConflictShareGraph::BuildImmutableOutAndNotSupportedRefreshInGraph() {
  DEF_GRAPH(g1) {
                  CHAIN(NODE("const", CONSTANT)->EDGE(0, 0)->NODE("swt", STREAMSWITCH));
                };
  auto graph = ToComputeGraph(g1);
  return graph;
}

/*
 *  const
 *    |
 *   swt
 *    |
 *    a
 */
ComputeGraphPtr MemConflictShareGraph::BuildImmutableOutAndNotSupportedRefreshOutByRefGraph() {
  DEF_GRAPH(g1) {
                  CHAIN(NODE("const", CONSTANT)->EDGE(0, 0)->NODE("swt", STREAMSWITCH)->NODE("a", RELU));
                };
  auto graph = ToComputeGraph(g1);
  SetOutReuseInput(graph, "swt");
  return graph;
}
/*
 *   const  swt
 *     \    /
 *   phonyconcat
 */
ComputeGraphPtr MemConflictShareGraph::BuildImmutableOutAndNotSupportedRefreshOutByContinuousInGraph() {
  DEF_GRAPH(g1) {
                  CHAIN(NODE("const", CONSTANT)->NODE("phonyconcat", PHONYCONCAT));
                  CHAIN(NODE("swt", STREAMSWITCH)->NODE("phonyconcat", PHONYCONCAT));
                };
  auto graph = ToComputeGraph(g1);
  SetNoPaddingContinuousInput(graph, "phonyconcat");
  return graph;
}
/*
 *   const
 *     |
 *   hcomallreduce (p2p input)
 */
ComputeGraphPtr MemConflictShareGraph::BuildImmutableOutAndRtsSpecailInGraph() {
  DEF_GRAPH(g1) {
                  CHAIN(NODE("const", CONSTANT)->EDGE(0, 0)->NODE("hcomallreduce", HCOMALLREDUCE));
                };
  std::vector<int64_t> in_mem_type {RT_MEMORY_P2P_DDR};
  auto graph = ToComputeGraph(g1);
  auto op1 = graph->FindNode("hcomallreduce");
  (void)ge::AttrUtils::SetListInt(op1->GetOpDescBarePtr(), ATTR_NAME_INPUT_MEM_TYPE_LIST, in_mem_type);
  return graph;
}
/*
 *    var  const
 *     \    /
 *     assign
 *       |
 *   hcombroadcast
 *       |
 *       a
 *       |
 *     netoutput
 */
ComputeGraphPtr MemConflictShareGraph::BuildImmutableOutAndRtsSpecailInByAssignGraph() {
  vector<int64_t> perm1(1 * 1 * 224 * 224, 0);
  GeTensorDesc tensor_desc1(GeShape(vector<int64_t>{1, 1, 224, 224}));
  GeTensorPtr const_tensor1 =
      std::make_shared<GeTensor>(tensor_desc1, reinterpret_cast<uint8_t *>(perm1.data()) , sizeof(int64_t)*perm1.size());
  auto constant = OP_CFG(CONSTANT).Weight(const_tensor1);
  auto hcombroadcast = OP_CFG(HCOMBROADCAST).Attr(ATTR_NAME_REFERENCE, true).InNames({"x"}).OutNames({"x"});
  const auto assign = OP_CFG(ASSIGN).Attr(ATTR_NAME_REFERENCE, true).InNames({"ref"}).OutNames({"ref"});

  DEF_GRAPH(g1) {
                  CHAIN(NODE("var", VARIABLE)->NODE("assign", assign)->NODE("hcombroadcast", hcombroadcast)
                            ->NODE("a", RELU)->NODE("netoutput", NETOUTPUT));
                  CHAIN(NODE("const", constant)->NODE("assign", assign));
                };
  auto graph = ToComputeGraph(g1);
  SetRtsSpecialTypeInput(graph, "hcombroadcast", RT_MEMORY_P2P_DDR);
  SetRtsSpecialTypeOutput(graph, "hcombroadcast", RT_MEMORY_P2P_DDR);

  // 如果不设置，会被设置为动态shape图
  AttrUtils::SetBool(graph, ATTR_NAME_NO_NEED_DYNAMIC_SHAPE_PARTITION, true);
  return graph;
}
/*
 *     data1
 *       |
 *       if
 *       |
 *      op3
 *       |
 *    netoutput
 *
 * then_subgraph          else_subgraph
 * +------------------+   +------------------+
 * | data2  netoutput1|   | data3  netoutput2|
 * |variable--+       |   |hcomallreduce--+  | hcomallreduce need p2p output
 * +------------------+   +------------------+
 *
 */
ComputeGraphPtr MemConflictShareGraph::BuildImmutableOutAndRtsSpecailOutGraph() {
  const auto data2 = OP_CFG(DATA).ParentNodeIndex(0);
  DEF_GRAPH(then_sub) {
                        CHAIN(NODE("variable", VARIABLE)->NODE("netoutput1", NETOUTPUT));
                        CHAIN(NODE("data2", data2));
                      };
  const auto data3 = OP_CFG(DATA).ParentNodeIndex(0);
  DEF_GRAPH(else_sub) {
                        CHAIN(NODE("data3", data3));
                        CHAIN(NODE("hcomallreduce", HCOMALLREDUCE)->NODE("netoutput2", NETOUTPUT));
                      };

  DEF_GRAPH(g1) {
                  CHAIN(NODE("data1", DATA)->EDGE(0, 0)->NODE("if", IF, then_sub, else_sub)
                            ->EDGE(0, 0)->NODE("op3", ADD)->EDGE(0, 0)->NODE("netoutput0", NETOUTPUT));
                };
  auto graph = ToComputeGraph(g1);
  SetRtsSpecialTypeOutput(graph, "hcomallreduce", RT_MEMORY_P2P_DDR);
  AddParentIndexForSubGraphNetoutput(graph);
  return graph;
}
/*
 *    var  const
 *     \    /
 *     assign
 *       |
 *   hcombroadcast (need p2p in/out)
 *       / \
 *      a   b
 *      |   |
 *     netoutput
 */
ComputeGraphPtr MemConflictShareGraph::BuildImmutableOutAnRtsSpecailOutByAssignOutput() {
  vector<int64_t> perm1(1 * 1 * 224 * 224, 0);
  GeTensorDesc tensor_desc1(GeShape(vector<int64_t>{1, 1, 224, 224}));
  GeTensorPtr const_tensor1 =
      std::make_shared<GeTensor>(tensor_desc1, reinterpret_cast<uint8_t *>(perm1.data()) , sizeof(int64_t)*perm1.size());
  auto constant = OP_CFG(CONSTANT).Weight(const_tensor1);
  auto hcombroadcast = OP_CFG(HCOMBROADCAST).Attr(ATTR_NAME_REFERENCE, true).InNames({"x"}).OutNames({"x", "y"});
  const auto assign = OP_CFG(ASSIGN).Attr(ATTR_NAME_REFERENCE, true).InNames({"ref"}).OutNames({"ref"});

  DEF_GRAPH(g1) {
                  CHAIN(NODE("var", VARIABLE)->NODE("assign", assign)->NODE("hcombroadcast", hcombroadcast)
                            ->NODE("a", RELU)->NODE("netoutput", NETOUTPUT));
                  CHAIN(NODE("const", constant)->NODE("assign", assign));
                  CHAIN(NODE("hcombroadcast", hcombroadcast)->NODE("b", RELU)->NODE("netoutput", NETOUTPUT));
                };
  auto graph = ToComputeGraph(g1);
  SetRefSrcVarName(graph, "assign", "var");
  SetRtsSpecialTypeOutput(graph, "hcombroadcast", RT_MEMORY_P2P_DDR);
  SetRtsSpecialTypeInput(graph, "hcombroadcast", RT_MEMORY_P2P_DDR);

  // 如果不设置，会被设置为动态shape图
  AttrUtils::SetBool(graph, ATTR_NAME_NO_NEED_DYNAMIC_SHAPE_PARTITION, true);
  return graph;
}

/*
 *    var  const
 *     \    /
 *     assign   c
 *       |     /
 *   hcombroadcast (need p2p in/out, continuous in/out, out ref in)
 *       / \
 *      a   b
 *      |   |
 *     netoutput
 */
ComputeGraphPtr MemConflictShareGraph::BuildImmutableOutAnRtsSpecailOutContinuousInOutByAssignGraph() {
  vector<int64_t> perm1(1 * 1 * 224 * 224, 0);
  GeTensorDesc tensor_desc1(GeShape(vector<int64_t>{1, 1, 224, 224}));
  GeTensorPtr const_tensor1 =
      std::make_shared<GeTensor>(tensor_desc1, reinterpret_cast<uint8_t *>(perm1.data()), sizeof(int64_t)*perm1.size());
  auto constant = OP_CFG(CONSTANT).Weight(const_tensor1);
  auto hcombroadcast = OP_CFG(HCOMBROADCAST).Attr(ATTR_NAME_REFERENCE, true).InNames({"x", "y"}).OutNames({"x", "y"});
  const auto assign = OP_CFG(ASSIGN).Attr(ATTR_NAME_REFERENCE, true).InNames({"ref"}).OutNames({"ref"});

  DEF_GRAPH(g1) {
                  CHAIN(NODE("var", VARIABLE)->NODE("assign", assign)->NODE("hcombroadcast", hcombroadcast)
                            ->NODE("a", RELU)->NODE("netoutput", NETOUTPUT));
                  CHAIN(NODE("const", constant)->NODE("assign", assign));
                  CHAIN(NODE("hcombroadcast", hcombroadcast)->NODE("b", RELU)->NODE("netoutput", NETOUTPUT));
                  CHAIN(NODE("c", RELU)->NODE("hcombroadcast", hcombroadcast));
                };
  auto graph = ToComputeGraph(g1);
  SetRefSrcVarName(graph, "assign", "var");
  SetRtsSpecialTypeOutput(graph, "hcombroadcast", RT_MEMORY_P2P_DDR);
  SetRtsSpecialTypeInput(graph, "hcombroadcast", RT_MEMORY_P2P_DDR);
  SetContinuousOutput(graph, "hcombroadcast");
  SetContinuousInput(graph, "hcombroadcast");
  // 如果不设置，会被设置为动态shape图
  AttrUtils::SetBool(graph, ATTR_NAME_NO_NEED_DYNAMIC_SHAPE_PARTITION, true);
  return graph;
}

/*
 *    op1
 *     +--hcomallreduce (p2p input)
 *     |
 *   switch
 */
ComputeGraphPtr MemConflictShareGraph::BuildNotSupportRefreshInAndRtsSpecialInGraph() {
  DEF_GRAPH(g1) {
                  CHAIN(NODE("op1", ADD)->NODE("hcomallreduce", HCOMALLREDUCE));
                  CHAIN(NODE("op1", ADD)->EDGE(0, 0)->NODE("switch", STREAMSWITCH));
                };
  std::vector<int64_t> in_mem_type {RT_MEMORY_P2P_DDR};
  auto graph = ToComputeGraph(g1);
  auto op1 = graph->FindNode("hcomallreduce");
  (void)ge::AttrUtils::SetListInt(op1->GetOpDescBarePtr(), ATTR_NAME_INPUT_MEM_TYPE_LIST, in_mem_type);
  return graph;
}

/*
 *   hcomallreduce (p2p output)
 *     |
 *   swtich
 */
ComputeGraphPtr MemConflictShareGraph::BuildNotSupportRefreshInAndRtsSpecialOutGraph() {
  DEF_GRAPH(g1) {
                  CHAIN(NODE("hcomallreduce", HCOMALLREDUCE)->EDGE(0, 0)->NODE("switch", STREAMSWITCH));
                };
  std::vector<int64_t> mem_type {RT_MEMORY_P2P_DDR};
  auto graph = ToComputeGraph(g1);
  auto op1 = graph->FindNode("hcomallreduce");
  (void)ge::AttrUtils::SetListInt(op1->GetOpDescBarePtr(), ATTR_NAME_OUTPUT_MEM_TYPE_LIST, mem_type);
  return graph;
}

/*
 *    add
 *     |
 *   swtich
 */
ComputeGraphPtr MemConflictShareGraph::BuildNotSupportRefreshInAndNormalOutGraph() {
  DEF_GRAPH(g1) {
                  CHAIN(NODE("add", ADD)->EDGE(0, 0)->NODE("switch", STREAMSWITCH));
                };
  auto graph = ToComputeGraph(g1);
  return graph;
}

/*
 *    add
 *     |
 *   hcom
 */
ComputeGraphPtr MemConflictShareGraph::BuildPhysicalRefreshableInAndNormalOutGraph() {
  DEF_GRAPH(g1) {
                  CHAIN(NODE("add", ADD)->EDGE(0, 0)->NODE("hcom", HCOMALLREDUCE));
                };
  auto graph = ToComputeGraph(g1);
  SetEngine(graph, {"hcom"}, kEngineNameHccl);
  return graph;
}

/*
 * unknown graph
 *      op1
 *       |               known sub graph
 * partitioned_call      +-----------+
 *       |               |   data    |
 *      op2              |    |      |
 *                       |   add1    |
 *                       |    |      |
 *                       |streamswitch|
 *                       |    |      |
 *                       |   add2    |
 *                       |    |      |
 *                       | netoutput |
 *                       +-----------+
 */
ComputeGraphPtr MemConflictShareGraph::BuildNotSupportRefreshInAndNormalOutInKnowSubgraph() {
  const auto sub_data = OP_CFG(DATA).ParentNodeIndex(0);
  DEF_GRAPH(sub_1) {
                     CHAIN(NODE("data", sub_data)->NODE("add1", ADD)->EDGE(0, 0)
                               ->NODE("switch", STREAMSWITCH)->NODE("add2", ADD)->NODE("netoutput", NETOUTPUT));
                   };
  DEF_GRAPH(g1) {
                  CHAIN(NODE("op1", ADD)->NODE("partitioned_call", PARTITIONEDCALL, sub_1)->NODE("op2", ADD));
                };
  auto graph = ToComputeGraph(g1);
  graph->SetGraphUnknownFlag(true);
  return graph;
}

/*
 *  swtich
 *    |
 *  hcomallreduce p2p in
 */
ComputeGraphPtr MemConflictShareGraph::BuildNotSupportRefreshOutAndRtsSpecialInGraph() {
  DEF_GRAPH(g1) {
                  CHAIN(NODE("switch", STREAMSWITCH)->EDGE(0, 0)->NODE("hcomallreduce", HCOMALLREDUCE));
                };
  std::vector<int64_t> in_mem_type {RT_MEMORY_P2P_DDR};
  auto graph = ToComputeGraph(g1);
  auto op1 = graph->FindNode("hcomallreduce");
  (void)ge::AttrUtils::SetListInt(op1->GetOpDescBarePtr(), ATTR_NAME_INPUT_MEM_TYPE_LIST, in_mem_type);
  return graph;
}

/*
 * unknown graph
 *      op1
 *       |               known sub graph
 * partitioned_call      +-----------+
 *       |               |   switch  |
 *      op2              |   |  |    |
 *                       |   add     |
 *                       |    |      |
 *                       |  netoutput|
 *                       +-----------+
 */
ComputeGraphPtr MemConflictShareGraph::BuildNotSupportRefreshOutAnchorIndex1Graph() {
  const auto sub_data = OP_CFG(DATA).ParentNodeIndex(0);
  DEF_GRAPH(sub_1) {
                     CHAIN(NODE("switch", STREAMSWITCH)->EDGE(0, 0)->NODE("add", ADD));
                     CHAIN(NODE("switch", STREAMSWITCH)->EDGE(1, 1)->NODE("add", ADD)->NODE("netoutput", NETOUTPUT));
                   };
  DEF_GRAPH(g1) {
                  CHAIN(NODE("op1", ADD)->NODE("partitioned_call", PARTITIONEDCALL, sub_1)->NODE("op2", ADD));
                };
  auto graph = ToComputeGraph(g1);
  graph->SetGraphUnknownFlag(true);
  return graph;
}
/*
 *     data1
 *       |
 *       if
 *       |
 *      op3
 *       |
 *    netoutput
 *
 * then_subgraph          else_subgraph
 * +--------------------+   +------------------------+
 * | data2              |   | data3                  |
 * |switch--netoutput1  |   |hcomallreduce-netoutput2|  hcomallreduce need p2p output
 * +--------------------+   +------------------------+
 */
ComputeGraphPtr MemConflictShareGraph::BuildNotSupportRefreshOutAndRtsSpecailOutGraph() {
  const auto data2 = OP_CFG(DATA).ParentNodeIndex(0);
  DEF_GRAPH(then_sub) {
                        CHAIN(NODE("switch", STREAMSWITCH)->NODE("netoutput1", NETOUTPUT));
                        CHAIN(NODE("data2", data2));
                      };
  const auto data3 = OP_CFG(DATA).ParentNodeIndex(0);
  DEF_GRAPH(else_sub) {
                        CHAIN(NODE("data3", data3));
                        CHAIN(NODE("hcomallreduce", HCOMALLREDUCE)->NODE("netoutput2", NETOUTPUT));
                      };

  DEF_GRAPH(g1) {
                  CHAIN(NODE("data1", DATA)->EDGE(0, 0)->NODE("if", IF, then_sub, else_sub)
                            ->EDGE(0, 0)->NODE("op3", ADD)->EDGE(0, 0)->NODE("netoutput0", NETOUTPUT));
                };
  auto graph = ToComputeGraph(g1);
  SetRtsSpecialTypeOutput(graph, "hcomallreduce", RT_MEMORY_P2P_DDR);
  AddParentIndexForSubGraphNetoutput(graph);
  return graph;
}

/*
 *     data1
 *       |
 *       if
 *       |
 *      op3
 *       |
 *    netoutput
 *
 * then_subgraph          else_subgraph
 * +--------------------+   +---------------+
 * | data2              |   | data3         |
 * |switch--netoutput1  |   |add-netoutput2 |
 * +--------------------+   +---------------+
 */
ComputeGraphPtr MemConflictShareGraph::BuildNotSupportRefreshOutAndNormalOutGraph() {
  const auto data2 = OP_CFG(DATA).ParentNodeIndex(0);
  DEF_GRAPH(then_sub) {
                        CHAIN(NODE("switch", STREAMSWITCH)->NODE("netoutput1", NETOUTPUT));
                        CHAIN(NODE("data2", data2));
                      };
  const auto data3 = OP_CFG(DATA).ParentNodeIndex(0);
  DEF_GRAPH(else_sub) {
                        CHAIN(NODE("data3", data3));
                        CHAIN(NODE("add", ADD)->NODE("netoutput2", NETOUTPUT));
                      };

  auto sub_graph = ToComputeGraph(else_sub);
  DEF_GRAPH(g1) {
                  CHAIN(NODE("data1", DATA)->EDGE(0, 0)->NODE("if", IF, then_sub, else_sub)
                            ->EDGE(0, 0)->NODE("op3", ADD)->EDGE(0, 0)->NODE("netoutput0", NETOUTPUT));
                };
  auto graph = ToComputeGraph(g1);
  AddParentIndexForSubGraphNetoutput(graph);
  return graph;
}
/*
 *     data1
 *       |
 *       if
 *      / \
 *      op3
 *       |
 *    netoutput
 *
 * then_subgraph          else_subgraph
 * +--------------------+   +---------------+
 * |data2               |   |data3          |
 * |switch+--netoutput1 |   |add--netoutput2|
 * |      +---+         |   |add2--+        |
 * +--------------------+   +---------------+
 */
ComputeGraphPtr MemConflictShareGraph::BuildNotSupportRefreshOutMultiReferenceAndNormalOutGraph() {
  const auto data2 = OP_CFG(DATA).ParentNodeIndex(0);
  DEF_GRAPH(then_sub) {
                        CHAIN(NODE("switch", STREAMSWITCH)->NODE("netoutput1", NETOUTPUT));
                        CHAIN(NODE("switch", STREAMSWITCH)->EDGE(0, 1)->NODE("netoutput1", NETOUTPUT));
                        CHAIN(NODE("data2", data2));
                      };
  const auto data3 = OP_CFG(DATA).ParentNodeIndex(0);
  DEF_GRAPH(else_sub) {
                        CHAIN(NODE("data3", data3));
                        CHAIN(NODE("add", ADD)->NODE("netoutput2", NETOUTPUT));
                        CHAIN(NODE("add2", ADD)->EDGE(0, 0)->NODE("netoutput2", NETOUTPUT));
                      };

  auto sub_graph = ToComputeGraph(else_sub);
  DEF_GRAPH(g1) {
                  CHAIN(NODE("data1", DATA)->EDGE(0, 0)->NODE("if", IF, then_sub, else_sub)
                            ->EDGE(0, 0)->NODE("op3", ADD)->NODE("netoutput0", NETOUTPUT));
                  CHAIN(NODE("if", IF, then_sub, else_sub)->NODE("op3", ADD));
                };
  auto graph = ToComputeGraph(g1);
  AddParentIndexForSubGraphNetoutput(graph);
  return graph;
}

/*
 *    op1
 *     |
 *    if           then_sub1         else_sub1
 *    /\           +--------------+  +--------------+
 *  op3 op4        | data1        |  | data1        |
 *   |   |         |  |           |  |              |
 *   netoutput     | memcpy1      |  | op2          |
 *                 |  |           |  |  |           |
 *                 | swt1    swt2 |  | memcpy2  op3 |
 *                 |  \       /   |  |  \       /   |
 *                 |   netoutput1 |  |   netoutput1 |
 *                 +--------------+  +--------------+
 */
ComputeGraphPtr MemConflictShareGraph::BuildNotSupportRefreshOutAndRtsSpecailIOAndNormalOutGraph() {
  const auto data1 = OP_CFG(DATA).ParentNodeIndex(0);
  DEF_GRAPH(then_sub1) {
                         CHAIN(NODE("data1", data1)->NODE("memcpy1", MEMCPYASYNC)->NODE("swt1", LABELSWITCHBYINDEX)
                                   ->NODE("netoutput1", NETOUTPUT));
                         CHAIN(NODE("swt2", LABELSWITCHBYINDEX)
                                   ->NODE("netoutput1", NETOUTPUT));
                       };
  const auto data2 = OP_CFG(DATA).ParentNodeIndex(0);
  DEF_GRAPH(else_sub1) {
                         CHAIN(NODE("data2", data2));
                         CHAIN(NODE("op2", RELU)->NODE("memcpy2", MEMCPYASYNC)->NODE("netoutput2", NETOUTPUT));
                         CHAIN(NODE("op3", RELU)->NODE("netoutput2", NETOUTPUT));
                       };

  DEF_GRAPH(g1) {
                  CHAIN(NODE("op1", RELU)->NODE("if", IF, then_sub1, else_sub1)
                            ->NODE("op4", RELU)->NODE("netoutput", NETOUTPUT));
                  CHAIN(NODE("if", IF, then_sub1, else_sub1)->NODE("op5", RELU));
                };

  auto graph = ToComputeGraph(g1);
  AddParentIndexForSubGraphNetoutput(graph);
  SetRtsSpecialTypeOutput(graph, "memcpy1", RT_MEMORY_TS);
  SetRtsSpecialTypeOutput(graph, "memcpy2", RT_MEMORY_TS);
  // 如果不设置，会被设置为动态shape图
  AttrUtils::SetBool(graph, ATTR_NAME_NO_NEED_DYNAMIC_SHAPE_PARTITION, true);
  return graph;
}
/*
 *  swt1   swt2
 *   \     /
 *    concat
 */
ComputeGraphPtr MemConflictShareGraph::BuildNotSupportRefreshOutAndContinuousInputGraph() {
  DEF_GRAPH(g1) {
                  CHAIN(NODE("swt1", STREAMSWITCH)->NODE("concat", CONCAT));
                  CHAIN(NODE("swt2", STREAMSWITCH)->NODE("concat", CONCAT));
                };
  auto graph = ToComputeGraph(g1);
  SetContinuousInput(graph, "concat");
  // 如果不设置，会被设置为动态shape图
  AttrUtils::SetBool(graph, ATTR_NAME_NO_NEED_DYNAMIC_SHAPE_PARTITION, true);
  return graph;
}
/*
 *  swt1   swt2
 *   \     /
 *  phony_concat
 */
ComputeGraphPtr MemConflictShareGraph::BuildNotSupportRefreshOutAndNoPaddingContinuousInputGraph() {
  DEF_GRAPH(g1) {
                  CHAIN(NODE("swt1", STREAMSWITCH)->NODE("phony_concat", PHONYCONCAT));
                  CHAIN(NODE("swt2", STREAMSWITCH)->NODE("phony_concat", PHONYCONCAT));
                };
  auto graph = ToComputeGraph(g1);
  SetNoPaddingContinuousInput(graph, "phony_concat");
  // 如果不设置，会被设置为动态shape图
  AttrUtils::SetBool(graph, ATTR_NAME_NO_NEED_DYNAMIC_SHAPE_PARTITION, true);
  return graph;
}

/*
 *    op1
 *     |
 *    if           then_sub1         else_sub1
 *    /\           +--------------+  +--------------+
 *  op3 op4        | data1        |  | data1        |
 *   |   |         |  |           |  |              |
 *  netoutput      | memcpy1      |  |              |
 *                 |  |           |  |              |
 *                 | swt1    swt2 |  | split        |
 *                 |  \       /   |  |  |  |        |
 *                 |   netoutput1 |  | netoutput1   |
 *                 +--------------+  +--------------+
 */
ComputeGraphPtr MemConflictShareGraph::BuildNotSupportRefreshOutAndContinuousOutputGraph() {
  const auto data1 = OP_CFG(DATA).ParentNodeIndex(0);
  DEF_GRAPH(then_sub1) {
                         CHAIN(NODE("data1", data1)->NODE("memcpy1", MEMCPYASYNC)->NODE("swt1", STREAMSWITCH)
                                   ->NODE("netoutput1", NETOUTPUT));
                         CHAIN(NODE("swt2", STREAMSWITCH)
                                   ->NODE("netoutput1", NETOUTPUT));
                       };
  const auto data2 = OP_CFG(DATA).ParentNodeIndex(0);
  DEF_GRAPH(else_sub1) {
                         CHAIN(NODE("data2", data2));
                         CHAIN(NODE("split", SPLIT)->NODE("netoutput2", NETOUTPUT));
                         CHAIN(NODE("split", SPLIT)->NODE("netoutput2", NETOUTPUT));
                       };

  DEF_GRAPH(g1) {
                  CHAIN(NODE("op1", RELU)->NODE("if", IF, then_sub1, else_sub1)
                            ->NODE("op4", RELU)->NODE("netoutput", NETOUTPUT));
                  CHAIN(NODE("if", IF, then_sub1, else_sub1)->NODE("op5", RELU));
                };

  auto graph = ToComputeGraph(g1);
  SetContinuousOutput(graph, "split");
  AddParentIndexForSubGraphNetoutput(graph);
  // 如果不设置，会被设置为动态shape图
  AttrUtils::SetBool(graph, ATTR_NAME_NO_NEED_DYNAMIC_SHAPE_PARTITION, true);
  return graph;
}

/*
 *    op1
 *     |
 *    if           then_sub1         else_sub1
 *    /\           +--------------+  +--------------+
 *  op3 op4        | data1        |  | data1        |
 *   |   |         |  |           |  |              |
 *  netoutput      | memcpy1      |  |              |
 *                 |  |           |  |              |
 *                 | swt1    swt2 |  | phony_split  |
 *                 |  \       /   |  |  |  |        |
 *                 |   netoutput1 |  | netoutput1   |
 *                 +--------------+  +--------------+
 */
ComputeGraphPtr MemConflictShareGraph::BuildNotSupportRefreshOutAndNoPaddingContinuousOutputGraph() {
  const auto data1 = OP_CFG(DATA).ParentNodeIndex(0);
  DEF_GRAPH(then_sub1) {
                         CHAIN(NODE("data1", data1)->NODE("memcpy1", MEMCPYASYNC)->NODE("swt1", STREAMSWITCH)
                                   ->NODE("netoutput1", NETOUTPUT));
                         CHAIN(NODE("swt2", STREAMSWITCH)
                                   ->NODE("netoutput1", NETOUTPUT));
                       };
  const auto data2 = OP_CFG(DATA).ParentNodeIndex(0);
  DEF_GRAPH(else_sub1) {
                         CHAIN(NODE("data2", data2));
                         CHAIN(NODE("phony_split", PHONYSPLIT)->NODE("netoutput2", NETOUTPUT));
                         CHAIN(NODE("phony_split", PHONYSPLIT)->NODE("netoutput2", NETOUTPUT));
                       };

  DEF_GRAPH(g1) {
                  CHAIN(NODE("op1", RELU)->NODE("if", IF, then_sub1, else_sub1)
                            ->NODE("op4", RELU)->NODE("netoutput", NETOUTPUT));
                  CHAIN(NODE("if", IF, then_sub1, else_sub1)->NODE("op5", RELU));
                };

  auto graph = ToComputeGraph(g1);
  SetNoPaddingContinuousOutput(graph, "phony_split");
  AddParentIndexForSubGraphNetoutput(graph);
  // 如果不设置，会被设置为动态shape图
  AttrUtils::SetBool(graph, ATTR_NAME_NO_NEED_DYNAMIC_SHAPE_PARTITION, true);
  return graph;
}
/*
 *    op1
 *     |
 *    if        then_sub1       else_sub1
 *     |        +------------+  +------------+
 *    op3       |   data1    |  | data1      |
 *     |        |    |       |  |            |
 *  netoutput   |   memcpy1  |  |            |
 *              |    |       |  |            |
 *              |   swt1     |  |    swt2    |
 *              |    |       |  |     |      |
 *              | netoutput1 |  | netoutput2 |
 *              +------------+  +------------+
 */
ComputeGraphPtr MemConflictShareGraph::BuildNotSupportRefreshOutAndNotSupportRefreshOutGraph() {
  const auto data1 = OP_CFG(DATA).ParentNodeIndex(0);
  DEF_GRAPH(then_sub1) {
                         CHAIN(NODE("data1", data1)->NODE("memcpy1", MEMCPYASYNC)->NODE("swt1", STREAMSWITCH)
                                   ->NODE("netoutput1", NETOUTPUT));
                       };
  const auto data2 = OP_CFG(DATA).ParentNodeIndex(0);
  DEF_GRAPH(else_sub1) {
                         CHAIN(NODE("data2", data2));
                         CHAIN(NODE("swt2", STREAMSWITCH)->NODE("netoutput2", NETOUTPUT));
                       };

  DEF_GRAPH(g1) {
                  CHAIN(NODE("op1", RELU)->NODE("if", IF, then_sub1, else_sub1)
                            ->NODE("op3", RELU)->NODE("netoutput", NETOUTPUT));
                };

  auto graph = ToComputeGraph(g1);
  AddParentIndexForSubGraphNetoutput(graph);
  // 如果不设置，会被设置为动态shape图
  AttrUtils::SetBool(graph, ATTR_NAME_NO_NEED_DYNAMIC_SHAPE_PARTITION, true);
  return graph;
}
/*
 *   data
 *    |
 *    a
 *    |
 *   swt1
 *    |
 *   swt2
 *    |
 *    b
 *    |
 *  netoutput
 */
ComputeGraphPtr MemConflictShareGraph::BuildNotSupportRefreshOutAndNotSupportRefreshOutByRefGraph() {
  DEF_GRAPH(g1) {
                  CHAIN(NODE("data", DATA)->NODE("a", RELU)->NODE("swt1", STREAMSWITCH)
                            ->NODE("swt2", STREAMSWITCH)->NODE("b", RELU)->NODE("netoutput", NETOUTPUT));
                };
  auto graph = ToComputeGraph(g1);
  SetOutReuseInput(graph, "swt2");
  // 如果不设置，会被设置为动态shape图
  AttrUtils::SetBool(graph, ATTR_NAME_NO_NEED_DYNAMIC_SHAPE_PARTITION, true);
  return graph;
}
/*
 *   data
 *    |
 *    a
 *    |
 *   swt1
 *    |
 *   swt2
 *    |
 *    b
 *    |
 *  netoutput
 */
ComputeGraphPtr MemConflictShareGraph::BuildNotSupportRefreshOutAndNotSupportRefreshInGraph() {
  DEF_GRAPH(g1) {
                  CHAIN(NODE("data", DATA)->NODE("a", RELU)->NODE("swt1", LABELSWITCHBYINDEX)
                            ->NODE("swt2", LABELSWITCHBYINDEX)->NODE("b", RELU)->NODE("netoutput", NETOUTPUT));
                };
  auto graph = ToComputeGraph(g1);
  // 如果不设置，会被设置为动态shape图
  AttrUtils::SetBool(graph, ATTR_NAME_NO_NEED_DYNAMIC_SHAPE_PARTITION, true);
  return graph;
}
/*
 *   a       b
 *   /\      /\
 *  | concat1  |
 *  \          /
 *    concat2
 */
ComputeGraphPtr MemConflictShareGraph::BuildContinuousInAndContinuousInGraph() {
  DEF_GRAPH(g1) {
                  CHAIN(NODE("a", RELU)->EDGE(0, 0)->NODE("concat1", CONCAT));
                  CHAIN(NODE("b", RELU)->EDGE(0, 1)->NODE("concat1", CONCAT));
                  CHAIN(NODE("a", RELU)->EDGE(0, 0)->NODE("concat2", CONCAT));
                  CHAIN(NODE("b", RELU)->EDGE(0, 1)->NODE("concat2", CONCAT));
                };
  auto graph = ToComputeGraph(g1);

  SetContinuousInput(graph, "concat1");
  SetContinuousInput(graph, "concat2");
  // 如果不设置，会被设置为动态shape图
  AttrUtils::SetBool(graph, ATTR_NAME_NO_NEED_DYNAMIC_SHAPE_PARTITION, true);
  return graph;
}
/*
 *   a       b
 *   \      /
 * c  concat1  d
 *  \   |      /
 *    concat2
 */
ComputeGraphPtr MemConflictShareGraph::BuildContinuousInAndContinuousInByRefGraph() {
  DEF_GRAPH(g1) {
                  CHAIN(NODE("a", RELU)->NODE("concat1", CONCAT));
                  CHAIN(NODE("b", RELU)->NODE("concat1", CONCAT));
                  CHAIN(NODE("c", RELU)->NODE("concat2", CONCAT));
                  CHAIN(NODE("concat1", CONCAT)->NODE("concat2", CONCAT));
                  CHAIN(NODE("d", RELU)->NODE("concat2", CONCAT));
                };
  auto graph = ToComputeGraph(g1);

  SetContinuousInput(graph, "concat1");
  SetOutReuseInput(graph, "concat1");
  SetContinuousInput(graph, "concat2");
  // 如果不设置，会被设置为动态shape图
  AttrUtils::SetBool(graph, ATTR_NAME_NO_NEED_DYNAMIC_SHAPE_PARTITION, true);
  return graph;
}
/*
 *     a     b
 *     /\    /\
 *    |  hcom  |
 *     \       /
 *   c  concat2  d
 *    \   |     /
 *      concat3
 *        |
 *      netoutput
 */
ComputeGraphPtr MemConflictShareGraph::BuildContinuousInAndContinuousInMixGraph() {
  DEF_GRAPH(g1) {
                  CHAIN(NODE("a", RELU)->EDGE(0, 0)->NODE("hcom", HCOMALLGATHER));
                  CHAIN(NODE("b", RELU)->EDGE(0, 1)->NODE("hcom", HCOMALLGATHER)->NODE("e", RELU)->NODE("netoutput", NETOUTPUT));
                  CHAIN(NODE("a", RELU)->EDGE(0, 0)->NODE("concat2", CONCAT));
                  CHAIN(NODE("b", RELU)->EDGE(0, 1)->NODE("concat2", CONCAT));

                  CHAIN(NODE("c", RELU)->NODE("concat3", CONCAT));
                  CHAIN(NODE("concat2", CONCAT)->NODE("concat3", CONCAT));
                  CHAIN(NODE("d", RELU)->NODE("concat3", CONCAT)->NODE("netoutput", NETOUTPUT));
                };
  auto graph = ToComputeGraph(g1);

  SetContinuousInput(graph, "hcom");
  SetContinuousInput(graph, "concat2");
  SetContinuousInput(graph, "concat3");
  SetOutReuseInput(graph, "concat2");
  // 如果不设置，会被设置为动态shape图
  AttrUtils::SetBool(graph, ATTR_NAME_NO_NEED_DYNAMIC_SHAPE_PARTITION, true);
  return graph;
}
/*
 *      a     b
 *      /\    /\
 *     |  hcom  |
 *      \       /
 *   c phonyconcat d
 *    \    |      /
 *       concat3
 *         |
 *      netoutput
 */
ComputeGraphPtr MemConflictShareGraph::BuildContinuousInAndNoPaddingContinuousInMixGraph() {
  DEF_GRAPH(g1) {
                  CHAIN(NODE("a", RELU)->EDGE(0, 0)->NODE("hcom", HCOMALLGATHER));
                  CHAIN(NODE("b", RELU)->EDGE(0, 1)->NODE("hcom", HCOMALLGATHER)->NODE("e", RELU)->NODE("netoutput", NETOUTPUT));
                  CHAIN(NODE("a", RELU)->EDGE(0, 0)->NODE("phonyconcat", PHONYCONCAT));
                  CHAIN(NODE("b", RELU)->EDGE(0, 1)->NODE("phonyconcat", PHONYCONCAT));

                  CHAIN(NODE("c", RELU)->NODE("concat3", CONCAT));
                  CHAIN(NODE("phonyconcat", PHONYCONCAT)->NODE("concat3", CONCAT));
                  CHAIN(NODE("d", RELU)->NODE("concat3", CONCAT)->NODE("netoutput", NETOUTPUT));
                };
  auto graph = ToComputeGraph(g1);

  auto a = graph->FindNode("a");
  a->GetOpDescBarePtr()->MutableOutputDesc(0)->SetShape(GeShape(std::vector<int64_t>({1, 1, 112, 448})));
  auto b = graph->FindNode("b");
  b->GetOpDescBarePtr()->MutableOutputDesc(0)->SetShape(GeShape(std::vector<int64_t>({1, 1, 112, 448})));

  auto pc = graph->FindNode("phonyconcat");
  pc->GetOpDescBarePtr()->MutableOutputDesc(0)->SetShape(GeShape(std::vector<int64_t>({1, 1, 224, 448})));

  SetContinuousInput(graph, "hcom");
  SetNoPaddingContinuousInput(graph, "phonyconcat");
  SetContinuousInput(graph, "concat3");

  // 如果不设置，会被设置为动态shape图
  AttrUtils::SetBool(graph, ATTR_NAME_NO_NEED_DYNAMIC_SHAPE_PARTITION, true);
  return graph;
}

/*
 *  a  split
 *  \   / \
 *  concat  b
 */
ComputeGraphPtr MemConflictShareGraph::BuildContinuousInAndContinuousOutGraph() {
  DEF_GRAPH(g1) {
                  CHAIN(NODE("a", RELU)->NODE("concat", CONCAT));
                  CHAIN(NODE("split", SPLIT)->NODE("concat", CONCAT)->NODE("netoutput", NETOUTPUT));
                  CHAIN(NODE("split", SPLIT)->NODE("b", RELU)->NODE("netoutput", NETOUTPUT));
                };
  auto graph = ToComputeGraph(g1);

  SetContinuousInput(graph, "concat");
  SetContinuousOutput(graph, "split");
  // 如果不设置，会被设置为动态shape图
  AttrUtils::SetBool(graph, ATTR_NAME_NO_NEED_DYNAMIC_SHAPE_PARTITION, true);
  return graph;
}

/*
 *  split
 *  /   \  |  /
 *       concat
 */
ComputeGraphPtr MemConflictShareGraph::BuildContinuousInHeadSameWithContinuousOutTailGraph() {
  DEF_GRAPH(g1) {
                  CHAIN(NODE("split", SPLIT)->NODE("a", RELU));
                  CHAIN(NODE("split", SPLIT)->NODE("concat", CONCAT));
                  CHAIN(NODE("b", RELU)->NODE("concat", CONCAT));
                  CHAIN(NODE("c", RELU)->NODE("concat", CONCAT));
                };
  auto graph = ToComputeGraph(g1);

  SetContinuousInput(graph, "concat");
  SetContinuousOutput(graph, "split");
  // 如果不设置，会被设置为动态shape图
  AttrUtils::SetBool(graph, ATTR_NAME_NO_NEED_DYNAMIC_SHAPE_PARTITION, true);
  return graph;
}

/*
 *          split
 *    \  |  /  \
 *      concat
 */
ComputeGraphPtr MemConflictShareGraph::BuildContinuousInTailSameWithContinuousOutHeadGraph() {
  DEF_GRAPH(g1) {
    CHAIN(NODE("a", RELU)->NODE("concat", CONCAT));
    CHAIN(NODE("b", RELU)->NODE("concat", CONCAT));
    CHAIN(NODE("split", SPLIT)->NODE("concat", CONCAT));
    CHAIN(NODE("split", SPLIT)->NODE("a", RELU));
  };
  auto graph = ToComputeGraph(g1);

  SetContinuousInput(graph, "concat");
  SetContinuousOutput(graph, "split");
  // 如果不设置，会被设置为动态shape图
  AttrUtils::SetBool(graph, ATTR_NAME_NO_NEED_DYNAMIC_SHAPE_PARTITION, true);
  return graph;
}

/*
 * 不冲突场景1：子集
 * node_a所有输出anchor     ： anchor1 anchor2 anchor3 anchor4
 * node_b所有输入anchor的对端：         anchor2 anchor3
 */
ComputeGraphPtr MemConflictShareGraph::BuildContinuousInIsSubSetOfContinuousOutGraph() {
  DEF_GRAPH(g1) {
                  CHAIN(NODE("split", SPLIT)->NODE("a", RELU));
                  CHAIN(NODE("split", SPLIT)->NODE("concat", CONCAT));
                  CHAIN(NODE("split", SPLIT)->NODE("concat", CONCAT));
                  CHAIN(NODE("split", SPLIT)->NODE("b", RELU));
                };
  auto graph = ToComputeGraph(g1);

  SetContinuousInput(graph, "concat");
  SetContinuousOutput(graph, "split");
  // 如果不设置，会被设置为动态shape图
  AttrUtils::SetBool(graph, ATTR_NAME_NO_NEED_DYNAMIC_SHAPE_PARTITION, true);
  return graph;
}

/*
 * 不冲突场景1：子集
 * node_a所有输出anchor     ：         anchor2 anchor3
 * node_b所有输入anchor的对端： anchor1 anchor2 anchor3 anchor4
 */
ComputeGraphPtr MemConflictShareGraph::BuildContinuousOutIsSubSetOfContinuousInGraph() {
  DEF_GRAPH(g1) {
    CHAIN(NODE("a", RELU)->NODE("concat", CONCAT));
    CHAIN(NODE("split", SPLIT)->NODE("concat", CONCAT));
    CHAIN(NODE("split", SPLIT)->NODE("concat", CONCAT));
    CHAIN(NODE("b", RELU)->NODE("concat", CONCAT));
  };
  auto graph = ToComputeGraph(g1);

  SetContinuousInput(graph, "concat");
  SetContinuousOutput(graph, "split");
  // 如果不设置，会被设置为动态shape图
  AttrUtils::SetBool(graph, ATTR_NAME_NO_NEED_DYNAMIC_SHAPE_PARTITION, true);
  return graph;
}

/*
 * 不冲突场景2：完全一样
 * node_a所有输出anchor     ： anchor1 anchor2 anchor3
 * node_b所有输入anchor的对端： anchor1 anchor2 anchor3
 */
ComputeGraphPtr MemConflictShareGraph::BuildContinuousInIsTheSameAsContinuousOutGraph() {
  DEF_GRAPH(g1) {
    CHAIN(NODE("split", SPLIT)->NODE("concat", CONCAT));
    CHAIN(NODE("split", SPLIT)->NODE("concat", CONCAT));
    CHAIN(NODE("split", SPLIT)->NODE("concat", CONCAT));
  };
  auto graph = ToComputeGraph(g1);

  SetContinuousInput(graph, "concat");
  SetContinuousOutput(graph, "split");
  // 如果不设置，会被设置为动态shape图
  AttrUtils::SetBool(graph, ATTR_NAME_NO_NEED_DYNAMIC_SHAPE_PARTITION, true);
  return graph;
}

/*
 * 冲突
 * node_a所有输出anchor     ： anchor1 anchor2
 * node_b所有输入anchor的对端：         anchor2 anchor1
 */
ComputeGraphPtr MemConflictShareGraph::BuildContinuousInReverseWithContinuousOutGraph() {
  DEF_GRAPH(g1) {
    CHAIN(NODE("split", SPLIT)->EDGE(0, 1)->NODE("concat", CONCAT));
    CHAIN(NODE("split", SPLIT)->EDGE(1, 0)->NODE("concat", CONCAT));
  };
  auto graph = ToComputeGraph(g1);

  SetContinuousInput(graph, "concat");
  SetContinuousOutput(graph, "split");
  // 如果不设置，会被设置为动态shape图
  AttrUtils::SetBool(graph, ATTR_NAME_NO_NEED_DYNAMIC_SHAPE_PARTITION, true);
  return graph;
}

/*
 * 冲突
 * node_a所有输出anchor     ：                 anchor1 anchor2
 * node_b所有输入anchor的对端：  anchor3 anchor2 anchor1
 */
ComputeGraphPtr MemConflictShareGraph::BuildContinuousInReverseWithContinuousOut2Graph() {
  DEF_GRAPH(g1) {
    CHAIN(NODE("a", RELU)->NODE("concat", CONCAT));
    CHAIN(NODE("split", SPLIT)->EDGE(0, 2)->NODE("concat", CONCAT));
    CHAIN(NODE("split", SPLIT)->EDGE(1, 1)->NODE("concat", CONCAT));
  };
  auto graph = ToComputeGraph(g1);

  SetContinuousInput(graph, "concat");
  SetContinuousOutput(graph, "split");
  // 如果不设置，会被设置为动态shape图
  AttrUtils::SetBool(graph, ATTR_NAME_NO_NEED_DYNAMIC_SHAPE_PARTITION, true);
  return graph;
}

/*
 * 冲突
 * node_a所有输出anchor     ： anchor1 anchor2 anchor3 anchor4
 * node_b所有输入anchor的对端：         anchor2         anchor4
 */
ComputeGraphPtr MemConflictShareGraph::BuildContinuousInAndContinuousOutPartialSame1Graph() {
  DEF_GRAPH(g1) {
    CHAIN(NODE("split", SPLIT)->NODE("a", RELU));
    CHAIN(NODE("split", SPLIT)->NODE("concat", CONCAT));
    CHAIN(NODE("split", SPLIT)->NODE("b", RELU));
    CHAIN(NODE("split", SPLIT)->NODE("concat", CONCAT));
  };
  auto graph = ToComputeGraph(g1);

  SetContinuousInput(graph, "concat");
  SetContinuousOutput(graph, "split");
  // 如果不设置，会被设置为动态shape图
  AttrUtils::SetBool(graph, ATTR_NAME_NO_NEED_DYNAMIC_SHAPE_PARTITION, true);
  return graph;
}

/*
 * 冲突
 * node_a所有输出anchor     ： anchor1 anchor2 anchor3 anchor4
 * node_b所有输入anchor的对端：         anchor2                 anchor5
 */
ComputeGraphPtr MemConflictShareGraph::BuildContinuousInAndContinuousOutPartialSame2Graph() {
  DEF_GRAPH(g1) {
    CHAIN(NODE("split", SPLIT)->NODE("a", RELU));
    CHAIN(NODE("split", SPLIT)->NODE("concat", CONCAT));
    CHAIN(NODE("split", SPLIT)->NODE("b", RELU));
    CHAIN(NODE("split", SPLIT)->NODE("c", RELU));
    CHAIN(NODE("d", RELU)->NODE("concat", CONCAT));
  };
  auto graph = ToComputeGraph(g1);

  SetContinuousInput(graph, "concat");
  SetContinuousOutput(graph, "split");
  // 如果不设置，会被设置为动态shape图
  AttrUtils::SetBool(graph, ATTR_NAME_NO_NEED_DYNAMIC_SHAPE_PARTITION, true);
  return graph;
}

/*
 * 冲突
 * node_a所有输出anchor     ： anchor1 anchor2 anchor3 anchor4
 * node_b所有输入anchor的对端：                                 anchor5  anchor2
 */
ComputeGraphPtr MemConflictShareGraph::BuildContinuousInAndContinuousOutPartialSame3Graph() {
  DEF_GRAPH(g1) {
    CHAIN(NODE("d", RELU)->NODE("concat", CONCAT));
    CHAIN(NODE("split", SPLIT)->NODE("a", RELU));
    CHAIN(NODE("split", SPLIT)->NODE("concat", CONCAT));
    CHAIN(NODE("split", SPLIT)->NODE("c", RELU));
    CHAIN(NODE("split", SPLIT)->NODE("d", RELU));
  };
  auto graph = ToComputeGraph(g1);

  SetContinuousInput(graph, "concat");
  SetContinuousOutput(graph, "split");
  // 如果不设置，会被设置为动态shape图
  AttrUtils::SetBool(graph, ATTR_NAME_NO_NEED_DYNAMIC_SHAPE_PARTITION, true);
  return graph;
}

/*
 * 冲突
 * node_a所有输出anchor     ：                anchor1 anchor2 anchor3 anchor4
 * node_b所有输入anchor的对端： anchor5 anchor6                anchor3 anchor4
 */
ComputeGraphPtr MemConflictShareGraph::BuildContinuousInAndContinuousOutPartialSame4Graph() {
  DEF_GRAPH(g1) {
    CHAIN(NODE("a", RELU)->NODE("concat", CONCAT));
    CHAIN(NODE("b", RELU)->NODE("concat", CONCAT));
    CHAIN(NODE("split", SPLIT)->NODE("c", RELU));
    CHAIN(NODE("split", SPLIT)->NODE("d", RELU));
    CHAIN(NODE("split", SPLIT)->NODE("concat", CONCAT));
    CHAIN(NODE("split", SPLIT)->NODE("concat", CONCAT));
  };
  auto graph = ToComputeGraph(g1);

  SetContinuousInput(graph, "concat");
  SetContinuousOutput(graph, "split");
  // 如果不设置，会被设置为动态shape图
  AttrUtils::SetBool(graph, ATTR_NAME_NO_NEED_DYNAMIC_SHAPE_PARTITION, true);
  return graph;
}
/*
 *   a       b
 *   /\      /\
 *  | concat   |
 *  \          /
 *   phony_concat
 */
ComputeGraphPtr MemConflictShareGraph::BuildContinuousInAndNoPaddingContinuousInGraph() {
  DEF_GRAPH(g1) {
                  CHAIN(NODE("a", RELU)->EDGE(0, 0)->NODE("concat", CONCAT));
                  CHAIN(NODE("b", RELU)->EDGE(0, 1)->NODE("concat", CONCAT));
                  CHAIN(NODE("a", RELU)->EDGE(0, 0)->NODE("phony_concat", PHONYCONCAT));
                  CHAIN(NODE("b", RELU)->EDGE(0, 1)->NODE("phony_concat", PHONYCONCAT));
                };
  auto graph = ToComputeGraph(g1);

  SetContinuousInput(graph, "concat");
  SetNoPaddingContinuousInput(graph, "phony_concat");
  // 如果不设置，会被设置为动态shape图
  AttrUtils::SetBool(graph, ATTR_NAME_NO_NEED_DYNAMIC_SHAPE_PARTITION, true);
  return graph;
}
/*
 *   a       b
 *   \      /
 * c  concat   d
 *  \   |      /
 *   phony_concat
 */
ComputeGraphPtr MemConflictShareGraph::BuildContinuousInAndNoPaddingContinuousInByRefGraph() {
  DEF_GRAPH(g1) {
                  CHAIN(NODE("a", RELU)->NODE("concat", CONCAT));
                  CHAIN(NODE("b", RELU)->NODE("concat", CONCAT));
                  CHAIN(NODE("c", RELU)->NODE("phony_concat", PHONYCONCAT));
                  CHAIN(NODE("concat", CONCAT)->NODE("phony_concat", PHONYCONCAT));
                  CHAIN(NODE("d", RELU)->NODE("phony_concat", PHONYCONCAT));
                };
  auto graph = ToComputeGraph(g1);

  SetContinuousInput(graph, "concat");
  SetOutReuseInput(graph, "concat");
  SetNoPaddingContinuousInput(graph, "phony_concat");
  // 如果不设置，会被设置为动态shape图
  AttrUtils::SetBool(graph, ATTR_NAME_NO_NEED_DYNAMIC_SHAPE_PARTITION, true);
  return graph;
}
/*
 *  a  phony_split
 *  \   / \
 *  concat  b
 */
ComputeGraphPtr MemConflictShareGraph::BuildContinuousInAndNoPaddingContinuousOutGraph() {
  DEF_GRAPH(g1) {
                  CHAIN(NODE("a", RELU)->NODE("concat", CONCAT));
                  CHAIN(NODE("phony_split", PHONYSPLIT)->NODE("concat", CONCAT));
                  CHAIN(NODE("phony_split", PHONYSPLIT)->NODE("b", RELU));
                };
  auto graph = ToComputeGraph(g1);

  SetContinuousInput(graph, "concat");
  SetNoPaddingContinuousOutput(graph, "phony_split");
  // 如果不设置，会被设置为动态shape图
  AttrUtils::SetBool(graph, ATTR_NAME_NO_NEED_DYNAMIC_SHAPE_PARTITION, true);
  return graph;
}
/*
 *   a   b
 *   \   /\
 *  concat hcom1
 */
ComputeGraphPtr MemConflictShareGraph::BuildContinuousInAndRtsSpecailInGraph() {
  DEF_GRAPH(g1) {
                  CHAIN(NODE("a", RELU)->NODE("concat", CONCAT));
                  CHAIN(NODE("b", RELU)->NODE("concat", CONCAT));
                  CHAIN(NODE("b", RELU)->EDGE(0, 0)->NODE("hcom1", HCOMALLREDUCE));
                };
  auto graph = ToComputeGraph(g1);

  SetContinuousInput(graph, "concat");
  SetRtsSpecialTypeInput(graph, "hcom1", RT_MEMORY_P2P_DDR);
  // 如果不设置，会被设置为动态shape图
  AttrUtils::SetBool(graph, ATTR_NAME_NO_NEED_DYNAMIC_SHAPE_PARTITION, true);
  return graph;
}
/*
 *   a    hcom1
 *     \  /
 *     concat
 */
ComputeGraphPtr MemConflictShareGraph::BuildContinuousInAndRtsSpecailOutGraph() {
  DEF_GRAPH(g1) {
                  CHAIN(NODE("a", RELU)->NODE("concat", CONCAT));
                  CHAIN(NODE("hcom1", HCOMALLREDUCE)->NODE("concat", CONCAT));
                };
  auto graph = ToComputeGraph(g1);

  SetContinuousInput(graph, "concat");
  SetRtsSpecialTypeOutput(graph, "hcom1", RT_MEMORY_P2P_DDR);
  // 如果不设置，会被设置为动态shape图
  AttrUtils::SetBool(graph, ATTR_NAME_NO_NEED_DYNAMIC_SHAPE_PARTITION, true);
  return graph;
}
/*
 *   a   b
 *    \ /
 *    hcom1
 *     |
 *   switch
 */
ComputeGraphPtr MemConflictShareGraph::BuildContinuousInAndRtsSpecailInByRefGraph() {
  auto hcom1 = OP_CFG(HCOMBROADCAST).Attr(ATTR_NAME_REFERENCE, true).InNames({"x"}).OutNames({"x"});
  DEF_GRAPH(g1) {
                  CHAIN(NODE("a", RELU)->NODE("hcom1", hcom1)->NODE("switch", STREAMSWITCH));
                  CHAIN(NODE("b", RELU)->NODE("hcom1", hcom1));
                };
  auto graph = ToComputeGraph(g1);

  SetContinuousInput(graph, "hcom1");
  SetRtsSpecialTypeInput(graph, "switch", RT_MEMORY_TS);
  // 如果不设置，会被设置为动态shape图
  AttrUtils::SetBool(graph, ATTR_NAME_NO_NEED_DYNAMIC_SHAPE_PARTITION, true);
  return graph;
}

/*
 *        a   b
 *         \ /
 *     c  PhonyConcat
 *     \    |
 *      hcom
 *      |   |
 *      d
 */
ComputeGraphPtr MemConflictShareGraph::BuildNoPaddingContinuousInAndRtsSpecailInByRefAndOutAnchorSuspendedGraph() {
  auto hcom1 = OP_CFG(HCOMBROADCAST).Attr(ATTR_NAME_REFERENCE, true).InNames({"x", "y"}).OutNames({"x", "y"});
  DEF_GRAPH(g1) {
                  CHAIN(NODE("c", RELU)->NODE("hcom", hcom1));
                  CHAIN(NODE("a", RELU)->NODE("pc", PHONYCONCAT)->NODE("hcom", hcom1)->NODE("d", RELU));
                  CHAIN(NODE("b", RELU)->NODE("pc", PHONYCONCAT));
                };
  auto graph = ToComputeGraph(g1);
  auto hcom = graph->FindNode("hcom");
  hcom->GetOpDescBarePtr()->AddOutputDesc(hcom->GetOpDescBarePtr()->GetInputDesc(0));

  SetNoPaddingContinuousInput(graph, "pc");
  SetOutReuseInput(graph, "pc");

  SetRtsSpecialTypeOutput(graph, "hcom", RT_MEMORY_P2P_DDR);
  // 如果不设置，会被设置为动态shape图
  AttrUtils::SetBool(graph, ATTR_NAME_NO_NEED_DYNAMIC_SHAPE_PARTITION, true);
  return graph;
}

/*
 *              data
 *               /\
 *  assign_slice0 assign_slice1 (inplace)
 *             \   /
 *              pc
 *              |
 *              b
 */
ComputeGraphPtr MemConflictShareGraph::BuildNoPaddingContinuousInWithSameAnchorData() {
  const auto assign = OP_CFG(ASSIGN).Attr(ATTR_NAME_REFERENCE, true).InNames({"ref"}).OutNames({"ref"});

  DEF_GRAPH(g1) {
                  CHAIN(NODE("data", DATA)->NODE("assign_slice0", assign)->NODE("pc", PHONYCONCAT));
                  CHAIN(NODE("data", DATA)->Data(0, 0)->NODE("assign_slice1", assign)->NODE("pc", PHONYCONCAT)->NODE("b", RELU));
                };
  auto graph = ToComputeGraph(g1);

  SetNoPaddingContinuousInput(graph, "pc");
  SetRefSrcVarName(graph, "assign_slice0", "var");
  SetRefSrcVarName(graph, "assign_slice1", "var");

  // 如果不设置，会被设置为动态shape图
  AttrUtils::SetBool(graph, ATTR_NAME_NO_NEED_DYNAMIC_SHAPE_PARTITION, true);
  return graph;
}

/*
 *              var
 *               /\
 *  assign_slice0 assign_slice1 (inplace)
 *             \   /
 *              pc
 *              |
 *              b
 */
ComputeGraphPtr MemConflictShareGraph::BuildNoPaddingContinuousInWithSameAnchorVariable() {
  const auto assign = OP_CFG(ASSIGN).Attr(ATTR_NAME_REFERENCE, true).InNames({"ref"}).OutNames({"ref"});

  DEF_GRAPH(g1) {
                  CHAIN(NODE("var", VARIABLE)->NODE("assign_slice0", assign)->NODE("pc", PHONYCONCAT));
                  CHAIN(NODE("var", VARIABLE)->Data(0, 0)->NODE("assign_slice1", assign)->NODE("pc", PHONYCONCAT)->NODE("b", RELU));
                };
  auto graph = ToComputeGraph(g1);

  SetNoPaddingContinuousInput(graph, "pc");
  SetRefSrcVarName(graph, "assign_slice0", "var");
  SetRefSrcVarName(graph, "assign_slice1", "var");

  // 如果不设置，会被设置为动态shape图
  AttrUtils::SetBool(graph, ATTR_NAME_NO_NEED_DYNAMIC_SHAPE_PARTITION, true);
  return graph;
}

/*
 *      b
 *      |
 *   a hcom1 c
 *    \ |   /\
 *     hcom2  f
 *      |      |
 *      d      e
 *       \    /
 *       netoutput
 *
 * hcom1: 需要p2p内存，且输出引用输入
 * hcom2:连续输入
 * c: 一个输出给2个人
 * f: 需要ts内存
 */
ComputeGraphPtr MemConflictShareGraph::BuildContinuousInAndRtsSpecailInOutGraph() {
  auto hcom1 = OP_CFG(HCOMBROADCAST).Attr(ATTR_NAME_REFERENCE, true).InNames({"x"}).OutNames({"x"});
  DEF_GRAPH(g1) {
                  CHAIN(NODE("a", RELU)->NODE("hcom2", HCOMALLGATHER)->NODE("d", RELU)->NODE("netoutput", NETOUTPUT));
                  CHAIN(NODE("b", RELU)->NODE("hcom1", hcom1)->NODE("hcom2", HCOMALLGATHER));
                  CHAIN(NODE("c", RELU)->NODE("hcom2", HCOMALLGATHER));
                  CHAIN(NODE("c", RELU)->EDGE(0, 0)->NODE("f", RELU)->NODE("e", RELU)->NODE("netoutput", NETOUTPUT));
                };
  auto graph = ToComputeGraph(g1);

  SetRtsSpecialTypeInput(graph, "hcom1", RT_MEMORY_P2P_DDR);
  SetRtsSpecialTypeOutput(graph, "hcom1", RT_MEMORY_P2P_DDR);
  SetContinuousInput(graph, "hcom2");
  SetRtsSpecialTypeInput(graph, "f", RT_MEMORY_TS);
  // 如果不设置，会被设置为动态shape图
  AttrUtils::SetBool(graph, ATTR_NAME_NO_NEED_DYNAMIC_SHAPE_PARTITION, true);
  return graph;
}
/*
 *      b
 *      |
 *  a  hcom1
 *   \ /
 *   hcom2
 *    |
 *    c
 *    |
 *  netoutput
 *  b:ts mem type
 *  hcom1: out ref input, p2p input and output
 *  hcom2: continuous input, p2p input
 */
ComputeGraphPtr MemConflictShareGraph::BuildContinuousInAndRtsSpecailInOutSameMemTypeGraph() {
  auto hcom1 = OP_CFG(HCOMBROADCAST).Attr(ATTR_NAME_REFERENCE, true).InNames({"x"}).OutNames({"x"});
  DEF_GRAPH(g1) {
                  CHAIN(NODE("a", RELU)->NODE("hcom2", HCOMALLGATHER)->NODE("c", RELU)->NODE("netoutput", NETOUTPUT));
                  CHAIN(NODE("b", RELU)->NODE("hcom1", hcom1)->NODE("hcom2", HCOMALLGATHER));
                };
  auto graph = ToComputeGraph(g1);

  SetRtsSpecialTypeInput(graph, "hcom1", RT_MEMORY_P2P_DDR);
  SetRtsSpecialTypeOutput(graph, "hcom1", RT_MEMORY_P2P_DDR);
  SetRtsSpecialTypeInput(graph, "hcom2", RT_MEMORY_P2P_DDR);
  SetContinuousInput(graph, "hcom2");
  SetRtsSpecialTypeOutput(graph, "b", RT_MEMORY_TS);
  // 如果不设置，会被设置为动态shape图
  AttrUtils::SetBool(graph, ATTR_NAME_NO_NEED_DYNAMIC_SHAPE_PARTITION, true);
  return graph;
}
/*
 *        split1
 *         /  \
 *     split2  c
 *     /   \
 *    a     b
 */
ComputeGraphPtr MemConflictShareGraph::BuildContinuousOutAndContinuousOutByRefGraph() {
  DEF_GRAPH(g1) {
                  CHAIN(NODE("split1", SPLIT)->NODE("split2", SPLIT));
                  CHAIN(NODE("split1", SPLIT)->NODE("c", RELU)->NODE("netoutput", NETOUTPUT));
                  CHAIN(NODE("split2", SPLIT)->NODE("a", RELU)->NODE("netoutput", NETOUTPUT));
                  CHAIN(NODE("split2", SPLIT)->NODE("b", RELU)->NODE("netoutput", NETOUTPUT));
                };
  auto graph = ToComputeGraph(g1);

  SetContinuousOutput(graph, "split1");
  SetContinuousOutput(graph, "split2");
  SetOutReuseInput(graph, "split2");
  // 如果不设置，会被设置为动态shape图
  AttrUtils::SetBool(graph, ATTR_NAME_NO_NEED_DYNAMIC_SHAPE_PARTITION, true);
  return graph;
}

/*
 *        split1
 *         /  \
 *     split2  c
 *     /   \
 *    a     b
 */
ComputeGraphPtr MemConflictShareGraph::BuildContinuousOutGraph() {
  DEF_GRAPH(g1) {
                  CHAIN(NODE("split1", SPLIT)->NODE("split2", SPLIT));
                  CHAIN(NODE("split1", SPLIT)->NODE("c", RELU)->NODE("netoutput", NETOUTPUT));
                  CHAIN(NODE("split2", SPLIT)->NODE("a", CAST)->NODE("netoutput", NETOUTPUT));
                  CHAIN(NODE("split2", SPLIT)->NODE("b", RELU)->NODE("netoutput", NETOUTPUT));
                };
  auto graph = ToComputeGraph(g1);

  SetContinuousOutput(graph, "split1");
  SetContinuousOutput(graph, "split2");
  // 如果不设置，会被设置为动态shape图
  AttrUtils::SetBool(graph, ATTR_NAME_NO_NEED_DYNAMIC_SHAPE_PARTITION, true);
  return graph;
}

/*
 *   var   hcom1
 *     \   /  \
 *     hcom2   c
 *     /   \
 *    a     b
 */
ComputeGraphPtr MemConflictShareGraph::BuildContinuousOutAndContinuousOutHcomByRefGraph() {
  auto hcombroadcast = OP_CFG(HCOMBROADCAST).Attr(ATTR_NAME_REFERENCE, true).InNames({"x1", "x2"}).OutNames({"x1", "x2"});
  DEF_GRAPH(g1) {
                  CHAIN(NODE("var", VARIABLE)->NODE("hcom2", hcombroadcast)->NODE("a", RELU)->NODE("netoutput", NETOUTPUT));
                  CHAIN(NODE("hcom1", HCOMALLGATHER)->NODE("hcom2", hcombroadcast)->NODE("b", RELU)->NODE("netoutput", NETOUTPUT));
                  CHAIN(NODE("hcom1", HCOMALLGATHER)->NODE("c", RELU)->NODE("netoutput", NETOUTPUT));
                };
  auto graph = ToComputeGraph(g1);

  SetContinuousOutput(graph, "hcom1");
  SetContinuousOutput(graph, "hcom2");
  // 如果不设置，会被设置为动态shape图
  AttrUtils::SetBool(graph, ATTR_NAME_NO_NEED_DYNAMIC_SHAPE_PARTITION, true);
  return graph;
}
/*
 *  split
 *  |  |
 * phony_concat
 */
ComputeGraphPtr MemConflictShareGraph::BuildContinuousOutAndNoPaddingContinuousInGraph() {
  DEF_GRAPH(g1) {
                  CHAIN(NODE("split", SPLIT)->NODE("phony_concat", PHONYCONCAT));
                  CHAIN(NODE("split", SPLIT)->NODE("phony_concat", PHONYCONCAT));
                };
  auto graph = ToComputeGraph(g1);

  SetContinuousOutput(graph, "split");
  SetNoPaddingContinuousInput(graph, "phony_concat");
  // 如果不设置，会被设置为动态shape图
  AttrUtils::SetBool(graph, ATTR_NAME_NO_NEED_DYNAMIC_SHAPE_PARTITION, true);
  return graph;
}
/*
 *        split
 *         /  \
 * phony_plit  c
 *     /   \
 *    a     b
 */
ComputeGraphPtr MemConflictShareGraph::BuildContinuousOutAndNoPaddingContinuousOutByRefGraph() {
  DEF_GRAPH(g1) {
                  CHAIN(NODE("split", SPLIT)->NODE("phony_plit", PHONYSPLIT));
                  CHAIN(NODE("split", SPLIT)->NODE("c", RELU)->NODE("netoutput", NETOUTPUT));
                  CHAIN(NODE("phony_plit", PHONYSPLIT)->NODE("a", RELU)->NODE("netoutput", NETOUTPUT));
                  CHAIN(NODE("phony_plit", PHONYSPLIT)->NODE("b", RELU)->NODE("netoutput", NETOUTPUT));
                };
  auto graph = ToComputeGraph(g1);

  auto split = graph->FindNode("split");
  split->GetOpDescBarePtr()->MutableOutputDesc(0)->SetShape(GeShape(std::vector<int64_t>({1, 1, 224, 448})));

  auto ps = graph->FindNode("phony_plit");
  ps->GetOpDescBarePtr()->MutableOutputDesc(0)->SetShape(GeShape(std::vector<int64_t>({1, 1, 112, 448})));
  ps->GetOpDescBarePtr()->MutableOutputDesc(1)->SetShape(GeShape(std::vector<int64_t>({1, 1, 112, 448})));

  SetContinuousOutput(graph, "split");
  SetNoPaddingContinuousOutput(graph, "phony_plit");
  // 如果不设置，会被设置为动态shape图
  AttrUtils::SetBool(graph, ATTR_NAME_NO_NEED_DYNAMIC_SHAPE_PARTITION, true);
  return graph;
}
/*
 *   a             b
 *   /\            /\
 *  | phony_concat1  |
 *  \                /
 *    phony_concat2
 */
ComputeGraphPtr MemConflictShareGraph::BuildNoPaddingContinuousInAndNoPaddingContinuousInGraph() {
  DEF_GRAPH(g1) {
                  CHAIN(NODE("a", RELU)->EDGE(0, 0)->NODE("phony_concat1", PHONYCONCAT));
                  CHAIN(NODE("b", RELU)->EDGE(0, 1)->NODE("phony_concat1", PHONYCONCAT));
                  CHAIN(NODE("a", RELU)->EDGE(0, 0)->NODE("phony_concat2", PHONYCONCAT));
                  CHAIN(NODE("b", RELU)->EDGE(0, 1)->NODE("phony_concat2", PHONYCONCAT));
                };
  auto graph = ToComputeGraph(g1);

  SetNoPaddingContinuousInput(graph, "phony_concat1");
  SetNoPaddingContinuousInput(graph, "phony_concat2");
  // 如果不设置，会被设置为动态shape图
  AttrUtils::SetBool(graph, ATTR_NAME_NO_NEED_DYNAMIC_SHAPE_PARTITION, true);
  return graph;
}

/*
 *    a    b   c
 *    |___|___|
 *        |
 *    d  pc1  f
 *    |___|___|
 *        |
 *       pc2
 */
ComputeGraphPtr MemConflictShareGraph::BuildNoPaddingContinuousInCascadedGraph() {
  DEF_GRAPH(g1) {
                  CHAIN(NODE("a", RELU)->EDGE(0, 0)->NODE("pc1", PHONYCONCAT));
                  CHAIN(NODE("b", RELU)->EDGE(0, 1)->NODE("pc1", PHONYCONCAT));
                  CHAIN(NODE("c", RELU)->EDGE(0, 2)->NODE("pc1", PHONYCONCAT));
                  CHAIN(NODE("d", RELU)->EDGE(0, 0)->NODE("pc2", PHONYCONCAT));
                  CHAIN(NODE("pc1", RELU)->EDGE(0, 1)->NODE("pc2", PHONYCONCAT));
                  CHAIN(NODE("f", RELU)->EDGE(0, 2)->NODE("pc2", PHONYCONCAT));
                };
  auto graph = ToComputeGraph(g1);

  SetNoPaddingContinuousInput(graph, "pc1");
  SetNoPaddingContinuousInput(graph, "pc2");
  // 如果不设置，会被设置为动态shape图
  AttrUtils::SetBool(graph, ATTR_NAME_NO_NEED_DYNAMIC_SHAPE_PARTITION, true);
  return graph;
}
/*
 *     a b c d      c d
 *     | | | |      | |
 *   phonyconcat1 phonyconcat2
 */
ComputeGraphPtr MemConflictShareGraph::BuildNoPaddingContinuousInAndNoPaddingContinuousInPartialSameInputsGraph() {
  DEF_GRAPH(g1) {
                  CHAIN(NODE("a", RELU)->NODE("phony_concat1", PHONYCONCAT));
                  CHAIN(NODE("b", RELU)->NODE("phony_concat1", PHONYCONCAT));
                  CHAIN(NODE("c", RELU)->NODE("phony_concat1", PHONYCONCAT));
                  CHAIN(NODE("d", RELU)->NODE("phony_concat1", PHONYCONCAT));
                  CHAIN(NODE("c", RELU)->EDGE(0, 0)->NODE("phony_concat2", PHONYCONCAT));
                  CHAIN(NODE("d", RELU)->EDGE(0, 1)->NODE("phony_concat2", PHONYCONCAT));
                };
  auto graph = ToComputeGraph(g1);

  SetNoPaddingContinuousInput(graph, "phony_concat1");
  SetNoPaddingContinuousInput(graph, "phony_concat2");
  // 如果不设置，会被设置为动态shape图
  AttrUtils::SetBool(graph, ATTR_NAME_NO_NEED_DYNAMIC_SHAPE_PARTITION, true);
  return graph;
}
/*
 *     a b c d      a d
 *     | | | |      | |
 *   phonyconcat1 phonyconcat2
 */
ComputeGraphPtr MemConflictShareGraph::BuildNoPaddingContinuousInAndNoPaddingContinuousInPartialSameInputsConflictGraph() {
  DEF_GRAPH(g1) {
                  CHAIN(NODE("a", RELU)->NODE("phony_concat1", PHONYCONCAT));
                  CHAIN(NODE("b", RELU)->NODE("phony_concat1", PHONYCONCAT));
                  CHAIN(NODE("c", RELU)->NODE("phony_concat1", PHONYCONCAT));
                  CHAIN(NODE("d", RELU)->NODE("phony_concat1", PHONYCONCAT)->NODE("e", RELU)->NODE("netoutput", NETOUTPUT));
                  CHAIN(NODE("a", RELU)->EDGE(0, 0)->NODE("phony_concat2", PHONYCONCAT));
                  CHAIN(NODE("d", RELU)->EDGE(0, 1)->NODE("phony_concat2", PHONYCONCAT)->NODE("e", RELU)->NODE("netoutput", NETOUTPUT));
                };
  auto graph = ToComputeGraph(g1);
  auto pc1 = graph->FindNode("phony_concat1");
  pc1->GetOpDescBarePtr()->MutableOutputDesc(0)->SetShape(GeShape(std::vector<int64_t>({1, 1, 224, 448})));
  auto pc2 = graph->FindNode("phony_concat2");
  pc2->GetOpDescBarePtr()->MutableOutputDesc(0)->SetShape(GeShape(std::vector<int64_t>({1, 1, 224, 448})));

  auto a = graph->FindNode("a");
  a->GetOpDescBarePtr()->MutableOutputDesc(0)->SetShape(GeShape(std::vector<int64_t>({1, 1, 56, 448})));
  auto b = graph->FindNode("b");
  b->GetOpDescBarePtr()->MutableOutputDesc(0)->SetShape(GeShape(std::vector<int64_t>({1, 1, 56, 448})));
  auto c = graph->FindNode("c");
  c->GetOpDescBarePtr()->MutableOutputDesc(0)->SetShape(GeShape(std::vector<int64_t>({1, 1, 56, 448})));
  auto d = graph->FindNode("d");
  d->GetOpDescBarePtr()->MutableOutputDesc(0)->SetShape(GeShape(std::vector<int64_t>({1, 1, 56, 448})));

  SetNoPaddingContinuousInput(graph, "phony_concat1");
  SetNoPaddingContinuousInput(graph, "phony_concat2");
  // 如果不设置，会被设置为动态shape图
  AttrUtils::SetBool(graph, ATTR_NAME_NO_NEED_DYNAMIC_SHAPE_PARTITION, true);
  return graph;
}
/*
 *     a b c d      e f c d
 *     | | | |      | | | |
 *   phonyconcat1 phonyconcat2
 */
ComputeGraphPtr MemConflictShareGraph::BuildNoPaddingContinuousInAndNoPaddingContinuousInTailPartialSameInputsGraph() {
  DEF_GRAPH(g1) {
                  CHAIN(NODE("a", RELU)->NODE("phony_concat1", PHONYCONCAT));
                  CHAIN(NODE("b", RELU)->NODE("phony_concat1", PHONYCONCAT));
                  CHAIN(NODE("c", RELU)->NODE("phony_concat1", PHONYCONCAT));
                  CHAIN(NODE("d", RELU)->NODE("phony_concat1", PHONYCONCAT));
                  CHAIN(NODE("e", RELU)->NODE("phony_concat2", PHONYCONCAT));
                  CHAIN(NODE("f", RELU)->NODE("phony_concat2", PHONYCONCAT));
                  CHAIN(NODE("c", RELU)->EDGE(0, 2)->NODE("phony_concat2", PHONYCONCAT));
                  CHAIN(NODE("d", RELU)->EDGE(0, 3)->NODE("phony_concat2", PHONYCONCAT));
                };
  auto graph = ToComputeGraph(g1);

  SetNoPaddingContinuousInput(graph, "phony_concat1");
  SetNoPaddingContinuousInput(graph, "phony_concat2");
  // 如果不设置，会被设置为动态shape图
  AttrUtils::SetBool(graph, ATTR_NAME_NO_NEED_DYNAMIC_SHAPE_PARTITION, true);
  return graph;
}
/*
 *     a b c d      e b
 *     | | | |      | |
 *   phonyconcat1 phonyconcat2
 */
ComputeGraphPtr MemConflictShareGraph::BuildNoPaddingContinuousInAndNoPaddingContinuousInTailIntersectionGraph() {
  DEF_GRAPH(g1) {
                  CHAIN(NODE("a", RELU)->NODE("phony_concat1", PHONYCONCAT));
                  CHAIN(NODE("b", RELU)->NODE("phony_concat1", PHONYCONCAT));
                  CHAIN(NODE("c", RELU)->NODE("phony_concat1", PHONYCONCAT));
                  CHAIN(NODE("d", RELU)->NODE("phony_concat1", PHONYCONCAT));

                  CHAIN(NODE("e", RELU)->NODE("phony_concat2", PHONYCONCAT));
                  CHAIN(NODE("b", RELU)->EDGE(0, 1)->NODE("phony_concat2", PHONYCONCAT));
                };
  auto graph = ToComputeGraph(g1);

  SetNoPaddingContinuousInput(graph, "phony_concat1");
  SetNoPaddingContinuousInput(graph, "phony_concat2");
  // 如果不设置，会被设置为动态shape图
  AttrUtils::SetBool(graph, ATTR_NAME_NO_NEED_DYNAMIC_SHAPE_PARTITION, true);
  return graph;
}

/*
 *   a            b
 *   /\1         0/\
 *  | phony_concat1 |
 *  \0            1/
 *    phony_concat2
 */
ComputeGraphPtr MemConflictShareGraph::BuildNoPaddingContinuousInAndNoPaddingContinuousInCrossGraph() {
  DEF_GRAPH(g1) {
                  CHAIN(NODE("a", RELU)->EDGE(0, 1)->NODE("phony_concat1", PHONYCONCAT));
                  CHAIN(NODE("b", RELU)->EDGE(0, 0)->NODE("phony_concat1", PHONYCONCAT));
                  CHAIN(NODE("a", RELU)->EDGE(0, 0)->NODE("phony_concat2", PHONYCONCAT));
                  CHAIN(NODE("b", RELU)->EDGE(0, 1)->NODE("phony_concat2", PHONYCONCAT));
                  CHAIN(NODE("phony_concat1", PHONYCONCAT)->NODE("c", RELU));
                  CHAIN(NODE("phony_concat2", PHONYCONCAT)->NODE("c", RELU));
                };
  auto graph = ToComputeGraph(g1);

  auto a = graph->FindNode("a");
  a->GetOpDescBarePtr()->MutableOutputDesc(0)->SetShape(GeShape(std::vector<int64_t>({1, 1, 56, 448})));
  auto b = graph->FindNode("b");
  b->GetOpDescBarePtr()->MutableOutputDesc(0)->SetShape(GeShape(std::vector<int64_t>({1, 1, 56, 448})));
  auto pc1 = graph->FindNode("phony_concat1");
  pc1->GetOpDescBarePtr()->MutableOutputDesc(0)->SetShape(GeShape(std::vector<int64_t>({1, 1, 112, 448})));
  auto pc2 = graph->FindNode("phony_concat2");
  pc2->GetOpDescBarePtr()->MutableOutputDesc(0)->SetShape(GeShape(std::vector<int64_t>({1, 1, 112, 448})));

  SetNoPaddingContinuousInput(graph, "phony_concat1");
  SetNoPaddingContinuousInput(graph, "phony_concat2");
  // 如果不设置，会被设置为动态shape图
  AttrUtils::SetBool(graph, ATTR_NAME_NO_NEED_DYNAMIC_SHAPE_PARTITION, true);
  return graph;
}
/*
 *    a        b
 *     \      /
 * c  phony_concat1  d
 *  \      |        /
 *    phony_concat2
 */
ComputeGraphPtr MemConflictShareGraph::BuildNoPaddingContinuousInAndNoPaddingContinuousInByRefGraph() {
  DEF_GRAPH(g1) {
                  CHAIN(NODE("a", RELU)->NODE("phony_concat1", CONCAT));
                  CHAIN(NODE("b", RELU)->NODE("phony_concat1", CONCAT));
                  CHAIN(NODE("c", RELU)->NODE("phony_concat2", CONCAT));
                  CHAIN(NODE("phony_concat1", CONCAT)->NODE("phony_concat2", CONCAT));
                  CHAIN(NODE("d", RELU)->NODE("phony_concat2", CONCAT));
                };
  auto graph = ToComputeGraph(g1);

  SetNoPaddingContinuousInput(graph, "phony_concat1");
  SetNoPaddingContinuousInput(graph, "phony_concat2");
  // 如果不设置，会被设置为动态shape图
  AttrUtils::SetBool(graph, ATTR_NAME_NO_NEED_DYNAMIC_SHAPE_PARTITION, true);
  return graph;
}

/*
 *      a           b
 *      /\          /\
 *     | phonyconcat1 |
 *      \            /
 *    c  phonyconcat2  d
 *     \      |       /
 *       phonyconcat3
 *            |
 *         netoutput
 */
ComputeGraphPtr MemConflictShareGraph::BuildNoPaddingContinuousInAndNoPaddingContinuousInMixGraph() {
  DEF_GRAPH(g1) {
                  CHAIN(NODE("a", RELU)->EDGE(0, 0)->NODE("phonyconcat1", PHONYCONCAT));
                  CHAIN(NODE("b", RELU)->EDGE(0, 1)->NODE("phonyconcat1", PHONYCONCAT)
                            ->NODE("e", RELU)->NODE("netoutput", NETOUTPUT)); // 连a连netoutput，避免phonyconcat1被优化掉
                  CHAIN(NODE("a", RELU)->EDGE(0, 0)->NODE("phonyconcat2", PHONYCONCAT));
                  CHAIN(NODE("b", RELU)->EDGE(0, 1)->NODE("phonyconcat2", PHONYCONCAT));

                  CHAIN(NODE("c", RELU)->NODE("phonyconcat3", PHONYCONCAT));
                  CHAIN(NODE("phonyconcat2", PHONYCONCAT)->NODE("phonyconcat3", PHONYCONCAT));
                  CHAIN(NODE("d", RELU)->NODE("phonyconcat3", PHONYCONCAT)->NODE("netoutput", NETOUTPUT));
                };
  auto graph = ToComputeGraph(g1);

  auto a = graph->FindNode("a");
  a->GetOpDescBarePtr()->MutableOutputDesc(0)->SetShape(GeShape(std::vector<int64_t>({1, 1, 112, 448})));
  auto b = graph->FindNode("b");
  b->GetOpDescBarePtr()->MutableOutputDesc(0)->SetShape(GeShape(std::vector<int64_t>({1, 1, 112, 448})));
  auto c = graph->FindNode("c");
  c->GetOpDescBarePtr()->MutableOutputDesc(0)->SetShape(GeShape(std::vector<int64_t>({1, 1, 112, 448})));
  auto d = graph->FindNode("d");
  d->GetOpDescBarePtr()->MutableOutputDesc(0)->SetShape(GeShape(std::vector<int64_t>({1, 1, 112, 448})));

  auto pc1 = graph->FindNode("phonyconcat1");
  pc1->GetOpDescBarePtr()->MutableOutputDesc(0)->SetShape(GeShape(std::vector<int64_t>({1, 1, 224, 448})));
  auto pc2 = graph->FindNode("phonyconcat2");
  pc2->GetOpDescBarePtr()->MutableOutputDesc(0)->SetShape(GeShape(std::vector<int64_t>({1, 1, 224, 448})));
  auto pc3 = graph->FindNode("phonyconcat3");
  pc3->GetOpDescBarePtr()->MutableOutputDesc(0)->SetShape(GeShape(std::vector<int64_t>({1, 1, 448, 448})));

  SetNoPaddingContinuousInput(graph, "phonyconcat1");
  SetNoPaddingContinuousInput(graph, "phonyconcat2");
  SetNoPaddingContinuousInput(graph, "phonyconcat3");
  // 如果不设置，会被设置为动态shape图
  AttrUtils::SetBool(graph, ATTR_NAME_NO_NEED_DYNAMIC_SHAPE_PARTITION, true);
  return graph;
}

/*
 *  a         phony_split
 *   \         / \
 *  phony_concat  b
 */
ComputeGraphPtr MemConflictShareGraph::BuildNoPaddingContinuousInAndNoPaddingContinuousOutGraph() {
  DEF_GRAPH(g1) {
                  CHAIN(NODE("a", RELU)->NODE("phony_concat", CONCAT));
                  CHAIN(NODE("phony_split", SPLIT)->NODE("phony_concat", CONCAT));
                  CHAIN(NODE("phony_split", SPLIT)->NODE("b", RELU));
                };
  auto graph = ToComputeGraph(g1);

  SetNoPaddingContinuousInput(graph, "phony_concat");
  SetNoPaddingContinuousOutput(graph, "phony_split");
  // 如果不设置，会被设置为动态shape图
  AttrUtils::SetBool(graph, ATTR_NAME_NO_NEED_DYNAMIC_SHAPE_PARTITION, true);
  return graph;
}
/*
 *   a          b
 *    \         /\
 *   phonyconcat phonysplit
 *       |         / \
 *       c        d   e
 *        \      /   /
 *         netoutput
 */
ComputeGraphPtr MemConflictShareGraph::BuildNoPaddingContinuousInAndNoPaddingContinuousOutByRefGraph() {
  DEF_GRAPH(g1) {
                  CHAIN(NODE("a", RELU)->NODE("phonyconcat", PHONYCONCAT)->NODE("c", RELU)->NODE("netoutput", NETOUTPUT));
                  CHAIN(NODE("b", RELU)->NODE("phonyconcat", PHONYCONCAT));
                  CHAIN(NODE("b", RELU)->EDGE(0, 0)->NODE("phonysplit", PHONYSPLIT)->NODE("d", RELU)->NODE("netoutput", NETOUTPUT));
                  CHAIN(NODE("phonysplit", PHONYSPLIT)->NODE("e", RELU)->NODE("netoutput", NETOUTPUT));
                };
  auto graph = ToComputeGraph(g1);

  auto a = graph->FindNode("a");
  a->GetOpDescBarePtr()->MutableOutputDesc(0)->SetShape(GeShape(std::vector<int64_t>({1, 1, 112, 448})));
  auto b = graph->FindNode("b");
  b->GetOpDescBarePtr()->MutableOutputDesc(0)->SetShape(GeShape(std::vector<int64_t>({1, 1, 112, 448})));
  auto pc = graph->FindNode("phonyconcat");
  pc->GetOpDescBarePtr()->MutableOutputDesc(0)->SetShape(GeShape(std::vector<int64_t>({1, 1, 224, 448})));
  auto ps = graph->FindNode("phonysplit");
  ps->GetOpDescBarePtr()->MutableOutputDesc(0)->SetShape(GeShape(std::vector<int64_t>({1, 1, 56, 448})));
  ps->GetOpDescBarePtr()->MutableOutputDesc(1)->SetShape(GeShape(std::vector<int64_t>({1, 1, 56, 448})));

  SetNoPaddingContinuousInput(graph, "phonyconcat");
  SetNoPaddingContinuousOutput(graph, "phonysplit");
  // 如果不设置，会被设置为动态shape图
  AttrUtils::SetBool(graph, ATTR_NAME_NO_NEED_DYNAMIC_SHAPE_PARTITION, true);
  return graph;
}
/*
 *    a    b
 *     \   /
 *  PhonyConcat
 *       |
 *   PhonySplit
 *      /  \
 *     c    d
 */
ComputeGraphPtr MemConflictShareGraph::BuildNoPaddingContinuousInAndNoPaddingContinuousOutConnectGraph() {
  DEF_GRAPH(g1) {
                  CHAIN(NODE("a", RELU)->NODE("phonyconcat", PHONYCONCAT)->NODE("phonysplit", PHONYSPLIT)
                            ->NODE("c", RELU)->NODE("netoutput", NETOUTPUT));
                  CHAIN(NODE("b", RELU)->NODE("phonyconcat", PHONYCONCAT));
                  CHAIN(NODE("phonysplit", PHONYSPLIT)->NODE("d", RELU)->NODE("netoutput", NETOUTPUT));
                };
  auto graph = ToComputeGraph(g1);

  auto a = graph->FindNode("a");
  a->GetOpDescBarePtr()->MutableOutputDesc(0)->SetShape(GeShape(std::vector<int64_t>({1, 1, 112, 448})));
  auto b = graph->FindNode("b");
  b->GetOpDescBarePtr()->MutableOutputDesc(0)->SetShape(GeShape(std::vector<int64_t>({1, 1, 112, 448})));
  auto pc = graph->FindNode("phonyconcat");
  pc->GetOpDescBarePtr()->MutableOutputDesc(0)->SetShape(GeShape(std::vector<int64_t>({1, 1, 224, 448})));
  auto ps = graph->FindNode("phonysplit");
  ps->GetOpDescBarePtr()->MutableOutputDesc(0)->SetShape(GeShape(std::vector<int64_t>({1, 1, 112, 448})));
  ps->GetOpDescBarePtr()->MutableOutputDesc(1)->SetShape(GeShape(std::vector<int64_t>({1, 1, 112, 448})));

  SetNoPaddingContinuousInput(graph, "phonyconcat");
  SetNoPaddingContinuousOutput(graph, "phonysplit");
  // 如果不设置，会被设置为动态shape图
  AttrUtils::SetBool(graph, ATTR_NAME_NO_NEED_DYNAMIC_SHAPE_PARTITION, true);
  return graph;
}
/*
 *       d
 *       |
 *   phony_split1
 *         /  \
 * phony_plit2  c
 *     /   \
 *    a     b
 */
ComputeGraphPtr MemConflictShareGraph::BuildNoPaddingContinuousOutAndNoPaddingContinuousOutGraph() {
  DEF_GRAPH(g1) {
                  CHAIN(NODE("d", RELU)->NODE("phony_plit1", PHONYSPLIT)->NODE("phony_plit2", PHONYSPLIT));
                  CHAIN(NODE("phony_plit1", PHONYSPLIT)->NODE("c", RELU)->NODE("netoutput", NETOUTPUT));
                  CHAIN(NODE("phony_plit2", PHONYSPLIT)->NODE("a", RELU)->NODE("netoutput", NETOUTPUT));
                  CHAIN(NODE("phony_plit2", PHONYSPLIT)->NODE("b", RELU)->NODE("netoutput", NETOUTPUT));
                };
  auto graph = ToComputeGraph(g1);
  auto d = graph->FindNode("d");
  d->GetOpDescBarePtr()->MutableOutputDesc(0)->SetShape(GeShape(std::vector<int64_t>({1, 1, 224, 448})));
  auto ps1 = graph->FindNode("phony_plit1");
  ps1->GetOpDescBarePtr()->MutableOutputDesc(0)->SetShape(GeShape(std::vector<int64_t>({1, 1, 112, 448})));
  ps1->GetOpDescBarePtr()->MutableOutputDesc(1)->SetShape(GeShape(std::vector<int64_t>({1, 1, 112, 448})));
  auto ps2 = graph->FindNode("phony_plit2");
  ps2->GetOpDescBarePtr()->MutableOutputDesc(0)->SetShape(GeShape(std::vector<int64_t>({1, 1, 56, 448})));
  ps2->GetOpDescBarePtr()->MutableOutputDesc(1)->SetShape(GeShape(std::vector<int64_t>({1, 1, 56, 448})));

  SetNoPaddingContinuousOutput(graph, "phony_plit1");
  SetNoPaddingContinuousOutput(graph, "phony_plit2");
  // 如果不设置，会被设置为动态shape图
  AttrUtils::SetBool(graph, ATTR_NAME_NO_NEED_DYNAMIC_SHAPE_PARTITION, true);
  return graph;
}
/*
 *       c
 *       |
 * phony_plit
 *     /   \
 *    a     b
 */
ComputeGraphPtr MemConflictShareGraph::BuildNoPaddingContinuousOutAndNormalOutGraph() {
  DEF_GRAPH(g1) {
                  CHAIN(NODE("c", RELU)->NODE("phony_plit", PHONYSPLIT));
                  CHAIN(NODE("phony_plit", PHONYSPLIT)->NODE("a", RELU));
                  CHAIN(NODE("phony_plit", PHONYSPLIT)->NODE("b", RELU));
                };
  auto graph = ToComputeGraph(g1);

  SetNoPaddingContinuousOutput(graph, "phony_plit");
  // 如果不设置，会被设置为动态shape图
  AttrUtils::SetBool(graph, ATTR_NAME_NO_NEED_DYNAMIC_SHAPE_PARTITION, true);
  return graph;
}
/*
 *   op1  hcom1 op2
 *     \   /    / \
 *      | |   /    \
 *       swt1      hcom2
 *       swt1地址不可刷新，hcom1 hcom2 特殊内存类型P2P
 */
ComputeGraphPtr MemConflictShareGraph::BuildNotSupportRefreshInputAndOtherConflictTypeGraph() {
  DEF_GRAPH(g1) {
                  CHAIN(NODE("op1", RELU)->EDGE(0, 0)->NODE("swt", LABELSWITCHBYINDEX));
                  CHAIN(NODE("hcom1", HCOMBROADCAST)->EDGE(0, 1)->NODE("swt", LABELSWITCHBYINDEX));
                  CHAIN(NODE("op2", RELU)->EDGE(0, 2)->NODE("swt", LABELSWITCHBYINDEX));
                  CHAIN(NODE("op2", RELU)->EDGE(0, 0)->NODE("hcom2", HCOMBROADCAST));
                };
  auto graph = ToComputeGraph(g1);

  SetRtsSpecialTypeInput(graph, "hcom2", RT_MEMORY_P2P_DDR);
  SetRtsSpecialTypeOutput(graph, "hcom1", RT_MEMORY_P2P_DDR);
  // 如果不设置，会被设置为动态shape图
  AttrUtils::SetBool(graph, ATTR_NAME_NO_NEED_DYNAMIC_SHAPE_PARTITION, true);
  return graph;
}
/*
 *   swt1  swt3
 *    /\     /\
 * swt2 concat swt4
 *    \   |   /
 *    netoutput
 */
ComputeGraphPtr MemConflictShareGraph::BuildNotSupportRefreshInputAndContinuousInputGraph() {
  DEF_GRAPH(g1) {
                  CHAIN(NODE("swt1", LABELSWITCHBYINDEX)->EDGE(0, 0)->NODE("swt2", LABELSWITCHBYINDEX)->NODE("netoutput", NETOUTPUT));
                  CHAIN(NODE("swt1", LABELSWITCHBYINDEX)->EDGE(0, 0)->NODE("concat", CONCAT));
                  CHAIN(NODE("swt3", LABELSWITCHBYINDEX)->EDGE(0, 1)->NODE("concat", CONCAT)->NODE("netoutput", NETOUTPUT));
                  CHAIN(NODE("swt3", LABELSWITCHBYINDEX)->EDGE(0, 0)->NODE("swt4", LABELSWITCHBYINDEX)->NODE("netoutput", NETOUTPUT));
                };
  auto graph = ToComputeGraph(g1);
  SetContinuousInput(graph, "concat");
  return graph;
}
/*
 *    a            b
 *    /\           /\
 * swt2 phony_concat swt4
 */
ComputeGraphPtr MemConflictShareGraph::BuildNotSupportRefreshInputAndNoPaddingContinuousInputGraph() {
  DEF_GRAPH(g1) {
                  CHAIN(NODE("a", RELU)->EDGE(0, 0)->NODE("swt2", STREAMSWITCH));
                  CHAIN(NODE("a", RELU)->EDGE(0, 0)->NODE("phony_concat", PHONYCONCAT));
                  CHAIN(NODE("b", STREAMSWITCH)->EDGE(0, 1)->NODE("phony_concat", PHONYCONCAT));
                  CHAIN(NODE("b", STREAMSWITCH)->EDGE(0, 0)->NODE("swt4", STREAMSWITCH));
                };
  auto graph = ToComputeGraph(g1);
  SetNoPaddingContinuousInput(graph, "phony_concat");
  return graph;
}
/*
 *    split
 *     / \
 *  swt1  swt2
 */
ComputeGraphPtr MemConflictShareGraph::BuildNotSupportRefreshInputAndContinuousOutputGraph() {
  DEF_GRAPH(g1) {
                  CHAIN(NODE("split", SPLIT)->NODE("swt1", STREAMSWITCH));
                  CHAIN(NODE("split", SPLIT)->NODE("swt2", STREAMSWITCH));
                };
  auto graph = ToComputeGraph(g1);
  SetContinuousOutput(graph, "split");
  return graph;
}
/*
 *  phony_split
 *     / \
 *  swt1  swt2
 */
ComputeGraphPtr MemConflictShareGraph::BuildNotSupportRefreshInputAndNoPaddingContinuousOutputGraph() {
  DEF_GRAPH(g1) {
                  CHAIN(NODE("phony_split", PHONYSPLIT)->NODE("swt1", STREAMSWITCH));
                  CHAIN(NODE("phony_split", PHONYSPLIT)->NODE("swt2", STREAMSWITCH));
                };
  auto graph = ToComputeGraph(g1);
  SetNoPaddingContinuousOutput(graph, "phony_split");
  return graph;
}

/*
 *  refdata--hcom--netoutput
 */
ComputeGraphPtr MemConflictShareGraph::BuildNotSupportRefreshInputAndRefDataGraph() {
  const auto ref_data = OP_CFG(REFDATA).InCnt(1);
  DEF_GRAPH(g1) {
                  CHAIN(NODE("refdata", ref_data)->NODE("hcom", HCOMALLREDUCE)->NODE("netoutput", NETOUTPUT));
                };
  auto graph = ToComputeGraph(g1);
  SetEngine(graph, {"hcom"}, kEngineNameHccl);

  // 如果不设置，会被设置为动态shape图
  AttrUtils::SetBool(graph, ATTR_NAME_NO_NEED_DYNAMIC_SHAPE_PARTITION, true);
  return graph;
}
/*
 *            add
 *           /     \
 *  hcomallreduce rtMemCpyAsync
 *   (p2p in)       (RT_MEMORY_TS input)
 */
ComputeGraphPtr MemConflictShareGraph::BuildRtsSpecialInAndRtsSpecialInGraph() {
  DEF_GRAPH(g1) {
                  CHAIN(NODE("add", ADD)->EDGE(0, 0)->NODE("hcomallreduce", HCOMALLREDUCE));
                  CHAIN(NODE("add", ADD)->EDGE(0, 0)->NODE("rtMemCpyAsync", MEMCPYASYNC));
                };
  auto graph = ToComputeGraph(g1);
  {
    std::vector<int64_t> mem_type {RT_MEMORY_P2P_DDR};
    auto op1 = graph->FindNode("hcomallreduce");
    (void)ge::AttrUtils::SetListInt(op1->GetOpDescBarePtr(), ATTR_NAME_INPUT_MEM_TYPE_LIST, mem_type);
  }
  {
    std::vector<int64_t> mem_type {RT_MEMORY_TS};
    auto op1 = graph->FindNode("rtMemCpyAsync");
    (void)ge::AttrUtils::SetListInt(op1->GetOpDescBarePtr(), ATTR_NAME_INPUT_MEM_TYPE_LIST, mem_type);
  }
  return graph;
}

/*
*   a
 *  |
 * hcom1
 *  |
 *  b
*/
ComputeGraphPtr MemConflictShareGraph::BuildRtsSpecialInAndRtsSpecialInByRefGraph() {
  auto hcom1 = OP_CFG(HCOMBROADCAST).Attr(ATTR_NAME_REFERENCE, true).InNames({"x"}).OutNames({"x"});
  DEF_GRAPH(g1) {
                  CHAIN(NODE("a", RELU)->NODE("hcom1", hcom1)->NODE("b", RELU));
                };
  auto graph = ToComputeGraph(g1);
  SetRtsSpecialTypeInput(graph, "hcom1", RT_MEMORY_P2P_DDR);
  SetRtsSpecialTypeOutput(graph, "hcom1", RT_MEMORY_P2P_DDR);
  SetRtsSpecialTypeInput(graph, "b", RT_MEMORY_TS);
  return graph;
}
/*
 *            op1
 *           /     \
 *        hcom   rtMemCpyAsync
 *   (p2p in)       (RT_MEMORY_TS input)
 */
ComputeGraphPtr MemConflictShareGraph::BuildRtsSpecialInAndOtherTypeGraph() {
  DEF_GRAPH(g1) {
                  CHAIN(NODE("op1", RELU)->EDGE(0, 0)->NODE("hcom", HCOMBROADCAST));
                  CHAIN(NODE("op1", RELU)->EDGE(0, 0)->NODE("rtMemCpyAsync", MEMCPYASYNC));
                };
  auto graph = ToComputeGraph(g1);
  SetRtsSpecialTypeInput(graph, "hcom", RT_MEMORY_P2P_DDR);
  SetRtsSpecialTypeInput(graph, "rtMemCpyAsync", RT_MEMORY_TS);
  // 如果不设置，会被设置为动态shape图
  AttrUtils::SetBool(graph, ATTR_NAME_NO_NEED_DYNAMIC_SHAPE_PARTITION, true);
  return graph;
}

/*
 *            add
 *           /     \
 *  hcomallreduce hcomallreduce2
 *   (p2p in)       (p2p in)
 */
ComputeGraphPtr MemConflictShareGraph::BuildRtsSpecialInAndRtsSpecialInSameTypeGraph() {
  DEF_GRAPH(g1) {
                  CHAIN(NODE("add", ADD)->EDGE(0, 0)->NODE("hcomallreduce", HCOMALLREDUCE));
                  CHAIN(NODE("add", ADD)->EDGE(0, 0)->NODE("hcomallreduce2", HCOMALLREDUCE));
                };
  auto graph = ToComputeGraph(g1);
  {
    std::vector<int64_t> mem_type {RT_MEMORY_P2P_DDR};
    auto op1 = graph->FindNode("hcomallreduce");
    (void)ge::AttrUtils::SetListInt(op1->GetOpDescBarePtr(), ATTR_NAME_INPUT_MEM_TYPE_LIST, mem_type);
  }
  {
    std::vector<int64_t> mem_type {RT_MEMORY_P2P_DDR};
    auto op1 = graph->FindNode("hcomallreduce2");
    (void)ge::AttrUtils::SetListInt(op1->GetOpDescBarePtr(), ATTR_NAME_INPUT_MEM_TYPE_LIST, mem_type);
  }
  return graph;
}

/*
 *   StreamSwitch (RT_MEMORY_TS out)
 *      |
 *  hcomallreduce  (p2p in)
 */
ComputeGraphPtr MemConflictShareGraph::BuildRtsSpecialInAndRtsSpecialOutGraph() {
  DEF_GRAPH(g1) {
                  CHAIN(NODE("StreamSwitch", STREAMSWITCH)->EDGE(0, 0)->NODE("hcomallreduce", HCOMALLREDUCE));
                };
  auto graph = ToComputeGraph(g1);
  {
    std::vector<int64_t> mem_type {RT_MEMORY_P2P_DDR};
    auto op1 = graph->FindNode("hcomallreduce");
    (void)ge::AttrUtils::SetListInt(op1->GetOpDescBarePtr(), ATTR_NAME_INPUT_MEM_TYPE_LIST, mem_type);
  }
  {
    std::vector<int64_t> mem_type {RT_MEMORY_TS};
    auto op1 = graph->FindNode("StreamSwitch");
    (void)ge::AttrUtils::SetListInt(op1->GetOpDescBarePtr(), ATTR_NAME_OUTPUT_MEM_TYPE_LIST, mem_type);
  }
  return graph;
}
/*
 *   a
 *   |
 *  hcom1
 *   |
 *  hcom2
 */
ComputeGraphPtr MemConflictShareGraph::BuildRtsSpecialInAndRtsSpecialOutByRefGraph() {
  auto hcom1 = OP_CFG(HCOMBROADCAST).Attr(ATTR_NAME_REFERENCE, true).InNames({"x"}).OutNames({"x"});
  DEF_GRAPH(g1) {
                  CHAIN(NODE("a", RELU)->NODE("hcom1", hcom1)->NODE("hcom2", HCOMALLGATHER));
                };
  auto graph = ToComputeGraph(g1);
  SetRtsSpecialTypeOutput(graph, "a", RT_MEMORY_TS);

  SetRtsSpecialTypeInput(graph, "hcom1", RT_MEMORY_P2P_DDR);
  SetRtsSpecialTypeOutput(graph, "hcom1", RT_MEMORY_P2P_DDR);
  SetRtsSpecialTypeInput(graph, "hcom2", RT_MEMORY_P2P_DDR);
  return graph;
}

/*
 *   hcomallreduce1 (p2p out)
 *      |
 *  hcomallreduce2  (p2p in)
 */
ComputeGraphPtr MemConflictShareGraph::BuildRtsSpecialInAndRtsSpecialOutSameTypeGraph() {
  DEF_GRAPH(g1) {
                  CHAIN(NODE("hcomallreduce1", MEMCPYASYNC)->EDGE(0, 0)->NODE("hcomallreduce2", HCOMALLREDUCE));
                };
  auto graph = ToComputeGraph(g1);
  {
    std::vector<int64_t> mem_type {RT_MEMORY_P2P_DDR};
    auto op1 = graph->FindNode("hcomallreduce2");
    (void)ge::AttrUtils::SetListInt(op1->GetOpDescBarePtr(), ATTR_NAME_INPUT_MEM_TYPE_LIST, mem_type);
  }
  {
    std::vector<int64_t> mem_type {RT_MEMORY_P2P_DDR};
    auto op1 = graph->FindNode("hcomallreduce1");
    (void)ge::AttrUtils::SetListInt(op1->GetOpDescBarePtr(), ATTR_NAME_OUTPUT_MEM_TYPE_LIST, mem_type);
  }
  return graph;
}

/*
 *     add
 *      |
 *  hcomallreduce  (p2p in)
 */
ComputeGraphPtr MemConflictShareGraph::BuildRtsSpecialInAndNormalOutGraph() {
  DEF_GRAPH(g1) {
                  CHAIN(NODE("add", ADD)->EDGE(0, 0)->NODE("hcomallreduce", HCOMALLREDUCE));
                };
  auto graph = ToComputeGraph(g1);
  {
    std::vector<int64_t> mem_type {RT_MEMORY_P2P_DDR};
    auto op1 = graph->FindNode("hcomallreduce");
    (void)ge::AttrUtils::SetListInt(op1->GetOpDescBarePtr(), ATTR_NAME_INPUT_MEM_TYPE_LIST, mem_type);
  }
  return graph;
}

/*
 *     data1
 *       |
 *       if
 *       |
 *      op3
 *       |
 *    netoutput
 *
 * then_subgraph          else_subgraph
 * +---------------------------+   +------------------------+
 * | data2                     |   | data3                  |
 * |rtMemCpyAsync--netoutput1  |   |hcomallreduce-netoutput2|  hcomallreduce need p2p output
 * +---------------------------+   +------------------------+
 */
ComputeGraphPtr MemConflictShareGraph::BuildRtsSpecialOutAndRtsSpecialOutGraph() {
  const auto data2 = OP_CFG(DATA).ParentNodeIndex(0);
  DEF_GRAPH(then_sub) {
                        CHAIN(NODE("rtMemCpyAsync", MEMCPYASYNC)->NODE("netoutput1", NETOUTPUT));
                        CHAIN(NODE("data2", data2));
                      };
  const auto data3 = OP_CFG(DATA).ParentNodeIndex(0);
  DEF_GRAPH(else_sub) {
                        CHAIN(NODE("data3", data3));
                        CHAIN(NODE("hcomallreduce", HCOMALLREDUCE)->NODE("netoutput2", NETOUTPUT));
                      };

  DEF_GRAPH(g1) {
                  CHAIN(NODE("data1", DATA)->EDGE(0, 0)->NODE("if", IF, then_sub, else_sub)
                            ->EDGE(0, 0)->NODE("op3", ADD)->EDGE(0, 0)->NODE("netoutput0", NETOUTPUT));
                };
  auto graph = ToComputeGraph(g1);
  SetRtsSpecialTypeOutput(graph, "hcomallreduce", RT_MEMORY_P2P_DDR);
  SetRtsSpecialTypeOutput(graph, "rtMemCpyAsync", RT_MEMORY_TS);
  AddParentIndexForSubGraphNetoutput(graph);
  return graph;
}

/*
 *    a
 *    |
 *   hcom1 (p2p in/out)
 *    |
 *   hcom2 (p2p in/out, output ref input)
 *    |
 *    b
 *    |
 *  netoutput
 */
ComputeGraphPtr MemConflictShareGraph::BuildRtsSpecialOutAndRtsSpecialOutByRefGraph() {
  auto hcom = OP_CFG(HCOMBROADCAST).Attr(ATTR_NAME_REFERENCE, true).InNames({"x"}).OutNames({"x"});
  DEF_GRAPH(g1) {
                  CHAIN(NODE("a", RELU)->NODE("hcom1", hcom)
                            ->NODE("hcom2", hcom)->NODE("b", RELU)->NODE("netoutput0", NETOUTPUT));
                };
  auto graph = ToComputeGraph(g1);
  SetRtsSpecialTypeOutput(graph, "hcom1", RT_MEMORY_P2P_DDR);
  SetRtsSpecialTypeOutput(graph, "hcom2", RT_MEMORY_P2P_DDR);
  SetRtsSpecialTypeInput(graph, "hcom1", RT_MEMORY_P2P_DDR);
  SetRtsSpecialTypeInput(graph, "hcom2", RT_MEMORY_P2P_DDR);
  return graph;
}

void AddSubgraphInstance(ge::ComputeGraphPtr parent_graph, ge::ComputeGraphPtr sub_graph,
                         const int parent_node_index, const std::string &parent_node_name) {
  auto sub_graph_name = sub_graph->GetName();
  sub_graph->SetParentGraph(parent_graph);
  sub_graph->SetParentNode(parent_graph->FindNode(parent_node_name));
  parent_graph->FindNode(parent_node_name)->GetOpDesc()->AddSubgraphName(sub_graph_name);
  parent_graph->FindNode(parent_node_name)->GetOpDesc()->SetSubgraphInstanceName(parent_node_index, sub_graph_name);
  parent_graph->AddSubgraph(sub_graph_name, sub_graph);
}

void AddWhileDummyCond(ge::ComputeGraphPtr parent_graph, const std::string &parent_node_name) {
  auto cond_graph = []() {
    DEF_GRAPH(g) {
      CHAIN(NODE("data0", "Data")->NODE("netoutput0", "NetOutput"));
    };
    return ToComputeGraph(g);
  }();
  cond_graph->SetName("cond");

  ge::AttrUtils::SetInt(cond_graph->FindFirstNodeMatchType("Data")->GetOpDesc(), "index", 0);
  ge::AttrUtils::SetInt(cond_graph->FindFirstNodeMatchType("Data")->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, 0);
  ge::AttrUtils::SetInt(cond_graph->FindFirstNodeMatchType("NetOutput")->GetOpDesc(), "index", 0);
  ge::AttrUtils::SetInt(cond_graph->FindFirstNodeMatchType("NetOutput")->GetOpDesc()->MutableInputDesc(0), ge::ATTR_NAME_PARENT_NODE_INDEX, 0);

  AddSubgraphInstance(parent_graph, cond_graph, 0, parent_node_name);
}

void AddWhileBody(ge::ComputeGraphPtr parent_graph, const std::string &parent_node_name) {
  auto body_graph = []() {
    DEF_GRAPH(g) {
      CHAIN(NODE("data1", "Data")->NODE("netoutput1", "NetOutput"));
    };
    return ToComputeGraph(g);
  }();
  body_graph->SetName("body");

  ge::AttrUtils::SetInt(body_graph->FindFirstNodeMatchType("Data")->GetOpDesc(), "index", 0);
  ge::AttrUtils::SetInt(body_graph->FindFirstNodeMatchType("Data")->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, 0);
  ge::AttrUtils::SetInt(body_graph->FindFirstNodeMatchType("NetOutput")->GetOpDesc(), "index", 0);
  ge::AttrUtils::SetInt(body_graph->FindFirstNodeMatchType("NetOutput")->GetOpDesc()->MutableInputDesc(0), ge::ATTR_NAME_PARENT_NODE_INDEX, 0);

  AddSubgraphInstance(parent_graph, body_graph, 1, parent_node_name);
}

void AddWhileBodyWithOneNode(ge::ComputeGraphPtr parent_graph, const std::string &parent_node_name) {
  auto body_graph = []() {
    DEF_GRAPH(g) {
      CHAIN(NODE("data1", DATA)->NODE("while_mul1", MUL)->NODE("netoutput1", NETOUTPUT));
    };
    return ToComputeGraph(g);
  }();
  body_graph->SetName("body");

  ge::AttrUtils::SetInt(body_graph->FindFirstNodeMatchType("Data")->GetOpDesc(), "index", 0);
  ge::AttrUtils::SetInt(body_graph->FindFirstNodeMatchType("Data")->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, 0);
  ge::AttrUtils::SetInt(body_graph->FindFirstNodeMatchType("NetOutput")->GetOpDesc(), "index", 0);
  ge::AttrUtils::SetInt(body_graph->FindFirstNodeMatchType("NetOutput")->GetOpDesc()->MutableInputDesc(0), ge::ATTR_NAME_PARENT_NODE_INDEX, 0);

  AddSubgraphInstance(parent_graph, body_graph, 1, parent_node_name);
}

void AddWhileBodyWithTwoNode(ge::ComputeGraphPtr parent_graph, const std::string &parent_node_name) {
  auto body_graph = []() {
    DEF_GRAPH(g) {
      CHAIN(NODE("data1", DATA)->NODE("while_mul1", MUL)->NODE("while_mul2", MUL)->NODE("netoutput1", NETOUTPUT));
    };
    return ToComputeGraph(g);
  }();
  body_graph->SetName("body");

  ge::AttrUtils::SetInt(body_graph->FindFirstNodeMatchType("Data")->GetOpDesc(), "index", 0);
  ge::AttrUtils::SetInt(body_graph->FindFirstNodeMatchType("Data")->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, 0);
  ge::AttrUtils::SetInt(body_graph->FindFirstNodeMatchType("NetOutput")->GetOpDesc(), "index", 0);
  ge::AttrUtils::SetInt(body_graph->FindFirstNodeMatchType("NetOutput")->GetOpDesc()->MutableInputDesc(0), ge::ATTR_NAME_PARENT_NODE_INDEX, 0);

  AddSubgraphInstance(parent_graph, body_graph, 1, parent_node_name);
}

void AddWhileBodyTwoExchange(ge::ComputeGraphPtr parent_graph, const std::string &parent_node_name) {
  auto body_graph = []() {
    DEF_GRAPH(g) {
      CHAIN(NODE("data2", "Data")->EDGE(0, 0)->NODE("netoutput1", "NetOutput"));
      CHAIN(NODE("data1", "Data")->EDGE(0, 1)->NODE("netoutput1", "NetOutput"));
    };
    return ToComputeGraph(g);
  }();
  body_graph->SetName("body");

  ge::AttrUtils::SetInt(body_graph->FindNode("data1")->GetOpDesc(), "index", 0);
  ge::AttrUtils::SetInt(body_graph->FindNode("data1")->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, 0);
  ge::AttrUtils::SetInt(body_graph->FindNode("data2")->GetOpDesc(), "index", 1);
  ge::AttrUtils::SetInt(body_graph->FindNode("data2")->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, 1);
  ge::AttrUtils::SetInt(body_graph->FindFirstNodeMatchType("NetOutput")->GetOpDesc()->MutableInputDesc(0), ge::ATTR_NAME_PARENT_NODE_INDEX, 0);
  ge::AttrUtils::SetInt(body_graph->FindFirstNodeMatchType("NetOutput")->GetOpDesc()->MutableInputDesc(1), ge::ATTR_NAME_PARENT_NODE_INDEX, 1);
  AddSubgraphInstance(parent_graph, body_graph, 1, parent_node_name);
}

void AddWhileBodyTwoNoExchange(ge::ComputeGraphPtr parent_graph, const std::string &parent_node_name) {
  auto body_graph = []() {
    DEF_GRAPH(g) {
      CHAIN(NODE("data1", "Data")->EDGE(0, 0)->NODE("netoutput1", "NetOutput"));
      CHAIN(NODE("data2", "Data")->EDGE(0, 1)->NODE("netoutput1", "NetOutput"));
    };
    return ToComputeGraph(g);
  }();
  body_graph->SetName("body");

  ge::AttrUtils::SetInt(body_graph->FindNode("data1")->GetOpDesc(), "index", 0);
  ge::AttrUtils::SetInt(body_graph->FindNode("data1")->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, 0);
  ge::AttrUtils::SetInt(body_graph->FindNode("data2")->GetOpDesc(), "index", 1);
  ge::AttrUtils::SetInt(body_graph->FindNode("data2")->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, 1);
  ge::AttrUtils::SetInt(body_graph->FindFirstNodeMatchType("NetOutput")->GetOpDesc()->MutableInputDesc(0), ge::ATTR_NAME_PARENT_NODE_INDEX, 0);
  ge::AttrUtils::SetInt(body_graph->FindFirstNodeMatchType("NetOutput")->GetOpDesc()->MutableInputDesc(1), ge::ATTR_NAME_PARENT_NODE_INDEX, 1);
  AddSubgraphInstance(parent_graph, body_graph, 1, parent_node_name);
}

/*
 *    pred
 *     |
 *    cast   input
 *      \    /
 *        if
 *        |
 *    netoutput
 *
 * then_subgraph          else_subgraph
 * +------------------+   +------------------+
 * |data1--netoutput1 |   |data2--netoutput2 |
 * +------------------+   +------------------+
 */
ComputeGraphPtr MemConflictShareGraph::BuildIfGraph() {
  auto main_graph = []() {
    DEF_GRAPH(g) {
      CHAIN(NODE("pred", "Data")->NODE("cast", "Cast")->NODE("if", "If")->NODE("netOutput", "NetOutput"));
      CHAIN(NODE("input", "Data")->EDGE(0, 1)->NODE("if", "If"));
    };
    return ToComputeGraph(g);
  }();
  main_graph->SetName("main");
  ge::AttrUtils::SetInt(main_graph->FindNode("cast")->GetOpDesc(), "index", 0);
  ge::AttrUtils::SetInt(main_graph->FindNode("input")->GetOpDesc(), "index", 1);
  
  auto then_graph = []() {
    DEF_GRAPH(g) {
      CHAIN(NODE("data1", "Data")->NODE("netOutput1", "NetOutput"));
    };
    return ToComputeGraph(g);
  }();

  then_graph->SetName("then");
  ge::AttrUtils::SetInt(then_graph->FindFirstNodeMatchType("Data")->GetOpDesc(), "index", 0);
  ge::AttrUtils::SetInt(then_graph->FindFirstNodeMatchType("Data")->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, 1);
  ge::AttrUtils::SetInt(then_graph->FindFirstNodeMatchType("NetOutput")->GetOpDesc(), "index", 0);
  ge::AttrUtils::SetInt(then_graph->FindFirstNodeMatchType("NetOutput")->GetOpDesc()->MutableInputDesc(0), ge::ATTR_NAME_PARENT_NODE_INDEX, 0);

  auto else_graph = []() {
    DEF_GRAPH(g) {
      CHAIN(NODE("data2", "Data")->NODE("netOutput2", "NetOutput"));
    };
    return ToComputeGraph(g);
  }();
  else_graph->SetName("else");
  ge::AttrUtils::SetInt(else_graph->FindFirstNodeMatchType("Data")->GetOpDesc(), "index", 0);
  ge::AttrUtils::SetInt(else_graph->FindFirstNodeMatchType("Data")->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, 1);
  ge::AttrUtils::SetInt(else_graph->FindFirstNodeMatchType("NetOutput")->GetOpDesc(), "index", 0);
  ge::AttrUtils::SetInt(else_graph->FindFirstNodeMatchType("NetOutput")->GetOpDesc()->MutableInputDesc(0), ge::ATTR_NAME_PARENT_NODE_INDEX, 0);

  auto if_node = main_graph->FindFirstNodeMatchType("If");
  then_graph->SetParentGraph(main_graph);
  then_graph->SetParentNode(if_node);
  else_graph->SetParentGraph(main_graph);
  else_graph->SetParentNode(if_node);

  main_graph->AddSubgraph(then_graph);
  main_graph->AddSubgraph(else_graph);
  if_node->GetOpDesc()->AddSubgraphName("then");
  if_node->GetOpDesc()->AddSubgraphName("else");
  if_node->GetOpDesc()->SetSubgraphInstanceName(0, "then");
  if_node->GetOpDesc()->SetSubgraphInstanceName(1, "else");
  if_node->GetOpDesc()->AppendIrInput("cond", kIrInputRequired);
  if_node->GetOpDesc()->AppendIrInput("input", kIrInputDynamic);
  auto &names_indexes = if_node->GetOpDesc()->MutableAllInputName();
  names_indexes.clear();
  names_indexes["cast"] = 0;
  names_indexes["input"] = 1;
  auto net_output = main_graph->FindNode("netOutput");
  net_output->GetOpDesc()->SetSrcName({"if"});
  net_output->GetOpDesc()->SetSrcIndex({0});
  main_graph->TopologicalSorting();

  main_graph->SetGraphUnknownFlag(false);
  for (auto &sub_graph : main_graph->GetAllSubgraphs()) {
    sub_graph->SetGraphUnknownFlag(false);
  }
  AttrUtils::SetBool(main_graph, ATTR_NAME_NO_NEED_DYNAMIC_SHAPE_PARTITION, true);
  return main_graph;
}

/*
 *  unknow sub graph
 *     data
 *      |                  know sub graph
 *   partitioned_call1   +-----------------+
 *      |                |          data2  |
 *    netoutput          |  op1 op2        |
 *                       |   \  /          |      then_subgraph       else_subgraph
 *                       |    if           |     +-----------------+ +-----------------+
 *                       |     |           |     | data3  data4    | | data5  data6    |
 *                       |    op3          |     |          |      | |          |      |
 *                       |     |           |     |      netoutput2 | |      netoutput3 |
 *                       |   netoutput1    |     +-----------------+ +-----------------+
 *                       +-----------------+
 */
ComputeGraphPtr MemConflictShareGraph::BuildIfInKnowSubGraph() {
  const auto data3 = OP_CFG(DATA).ParentNodeIndex(0);
  const auto data4 = OP_CFG(DATA).ParentNodeIndex(1);
  DEF_GRAPH(then_sub) {
                        CHAIN(NODE("data4", data4)->NODE("netoutput2", NETOUTPUT));
                        CHAIN(NODE("data3", data3));
                      };
  const auto data5 = OP_CFG(DATA).ParentNodeIndex(0);
  const auto data6 = OP_CFG(DATA).ParentNodeIndex(1);
  DEF_GRAPH(else_sub) {
                        CHAIN(NODE("data5", data5));
                        CHAIN(NODE("data6", data6)->NODE("netoutput3", NETOUTPUT));
                      };

  const auto data2 = OP_CFG(DATA).ParentNodeIndex(0);
  DEF_GRAPH(partitioned_call1) {
                                 CHAIN(NODE("data2", data2));
                                 CHAIN(NODE("op1", RELU)->NODE("if", IF, then_sub, else_sub)
                                           ->NODE("op3", ADD)->NODE("netoutput1", NETOUTPUT));
                                 CHAIN(NODE("op2", RELU)->NODE("if", IF, then_sub, else_sub));
                };
  DEF_GRAPH(g1) {
                  CHAIN(NODE("data", DATA)->NODE("partitioned_call1", PARTITIONEDCALL, partitioned_call1)
                            ->NODE("netoutput", NETOUTPUT));
                };
  auto root_graph = ToComputeGraph(g1);

  auto then_sub_graph = ToComputeGraph(then_sub);
  auto else_sub_graph = ToComputeGraph(else_sub);
  auto p1 = root_graph->GetSubgraph("partitioned_call1");
  auto if_parent_node = p1->FindNode("if");
  if (if_parent_node == nullptr) {
    return nullptr;
  }
  then_sub_graph->SetParentNode(if_parent_node);
  then_sub_graph->SetParentGraph(root_graph);
  else_sub_graph->SetParentNode(if_parent_node);
  else_sub_graph->SetParentGraph(root_graph);
  root_graph->AddSubGraph(then_sub_graph);
  root_graph->AddSubGraph(else_sub_graph);

  AddParentIndexForSubGraphNetoutput(root_graph);
  root_graph->SetGraphUnknownFlag(true);
  return root_graph;
}

/*
 *    pred
 *     |
 *    cast   input
 *      \    /
 *        if
 *        \/
 *    netoutput
 *
 * then_subgraph        else_subgraph
 * +----------------+   +---------------+
 * |     data1      |   |    data2      |
 * |      |         |   |      |        |
 * |     mul1       |   |     / \       |
 * |      |         |   |   mu2  mu3    |
 * |     / \        |   |    |    |     |
 * |   netoutput1   |   |   netoutput2  |
 * +----------------+   +---------------+
 */
ComputeGraphPtr MemConflictShareGraph::BuildIfSingleOutMultiRefToNetoutputSubGraph() {
  auto main_graph = []() {
    DEF_GRAPH(g) {
      CHAIN(NODE("pred", DATA)->NODE("cast", CAST)->NODE("if", IF)->EDGE(0, 0)->NODE("netoutput", NETOUTPUT));
      CHAIN(NODE("input", DATA)->EDGE(0, 1)->NODE("if", IF)->EDGE(1, 1)->NODE("netoutput", NETOUTPUT));
    };
    return ToComputeGraph(g);
  }();
  main_graph->SetName("main");
  ge::AttrUtils::SetInt(main_graph->FindNode("cast")->GetOpDesc(), "index", 0);
  ge::AttrUtils::SetInt(main_graph->FindNode("input")->GetOpDesc(), "index", 1);
  
  auto then_graph = []() {
    DEF_GRAPH(g) {
      CHAIN(NODE("data1", DATA)->NODE("mul1", MUL)->EDGE(0, 0)->NODE("netoutput1", NETOUTPUT));
      CHAIN(NODE("mul1", MUL)->EDGE(0, 1)->NODE("netoutput1", NETOUTPUT));
    };
    return ToComputeGraph(g);
  }();

  then_graph->SetName("then");
  ge::AttrUtils::SetInt(then_graph->FindFirstNodeMatchType(DATA)->GetOpDesc(), "index", 0);
  ge::AttrUtils::SetInt(then_graph->FindFirstNodeMatchType(DATA)->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, 1);
  ge::AttrUtils::SetInt(then_graph->FindFirstNodeMatchType(NETOUTPUT)->GetOpDesc(), "index", 0);
  ge::AttrUtils::SetInt(then_graph->FindFirstNodeMatchType(NETOUTPUT)->GetOpDesc()->MutableInputDesc(0), ge::ATTR_NAME_PARENT_NODE_INDEX, 0);

  auto else_graph = []() {
    DEF_GRAPH(g) {
      CHAIN(NODE("data2", DATA)->EDGE(0, 0)->NODE("mul2", MUL)->NODE("netoutput2", NETOUTPUT));
      CHAIN(NODE("data2", DATA)->EDGE(0, 0)->NODE("mul3", MUL)->NODE("netoutput2", NETOUTPUT));
    };
    return ToComputeGraph(g);
  }();
  else_graph->SetName("else");
  ge::AttrUtils::SetInt(else_graph->FindFirstNodeMatchType(DATA)->GetOpDesc(), "index", 0);
  ge::AttrUtils::SetInt(else_graph->FindFirstNodeMatchType(DATA)->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, 1);
  ge::AttrUtils::SetInt(else_graph->FindFirstNodeMatchType(NETOUTPUT)->GetOpDesc(), "index", 0);
  ge::AttrUtils::SetInt(else_graph->FindFirstNodeMatchType(NETOUTPUT)->GetOpDesc()->MutableInputDesc(0), ge::ATTR_NAME_PARENT_NODE_INDEX, 0);

  auto if_node = main_graph->FindFirstNodeMatchType("If");
  then_graph->SetParentGraph(main_graph);
  then_graph->SetParentNode(if_node);
  else_graph->SetParentGraph(main_graph);
  else_graph->SetParentNode(if_node);
  main_graph->AddSubgraph(then_graph);
  main_graph->AddSubgraph(else_graph);
  if_node->GetOpDesc()->AddSubgraphName("then");
  if_node->GetOpDesc()->AddSubgraphName("else");
  if_node->GetOpDesc()->SetSubgraphInstanceName(0, "then");
  if_node->GetOpDesc()->SetSubgraphInstanceName(1, "else");
  if_node->GetOpDesc()->AppendIrInput("cond", kIrInputRequired);
  if_node->GetOpDesc()->AppendIrInput("input", kIrInputDynamic);
  auto &names_indexes = if_node->GetOpDesc()->MutableAllInputName();
  names_indexes.clear();
  names_indexes["cast"] = 0;
  names_indexes["input"] = 1;
  auto net_output = main_graph->FindNode("netoutput");
  net_output->GetOpDesc()->SetSrcName({"if"});
  net_output->GetOpDesc()->SetSrcIndex({0});
  main_graph->TopologicalSorting();

  main_graph->SetGraphUnknownFlag(false);
  for (auto &sub_graph : main_graph->GetAllSubgraphs()) {
    sub_graph->SetGraphUnknownFlag(false);
  }
  AttrUtils::SetBool(main_graph, ATTR_NAME_NO_NEED_DYNAMIC_SHAPE_PARTITION, true);
  return main_graph;
}

/*
 *    pred
 *     |
 *    cast   input
 *      \    /
 *        if
 *        \/
 *    netoutput
 *
 * then_subgraph                            else_subgraph
 * +------------------------------------+   +---------------+
 * |     data1                          |   |    data2      |
 * |      |                             |   |      |        |
 * |     mul1                           |   |     / \       |
 * |      |                             |   |   mu2  mu3    |
 * |     / \                            |   |    |    |     |
 * |    |  partitionedcall +----------+ |   |   netoutput2  |
 * |    |    |             |  data3   | |   +---------------+
 * |   netoutput1          |    |     | |
 * |                       |netoutput3| |
 * |                       +----------+ |
 * +------------------------------------+
 */
ComputeGraphPtr MemConflictShareGraph::BuildIfSingleOutMultiRefToNetoutputSubGraphWithPartitionedCall() {
  auto main_graph = []() {
    DEF_GRAPH(g) {
      CHAIN(NODE("pred", DATA)->NODE("cast", CAST)->NODE("if", IF)->EDGE(0, 0)->NODE("netoutput", NETOUTPUT));
      CHAIN(NODE("input", DATA)->EDGE(0, 1)->NODE("if", IF)->EDGE(1, 1)->NODE("netoutput", NETOUTPUT));
    };
    return ToComputeGraph(g);
  }();
  main_graph->SetName("main");
  ge::AttrUtils::SetInt(main_graph->FindNode("cast")->GetOpDesc(), "index", 0);
  ge::AttrUtils::SetInt(main_graph->FindNode("input")->GetOpDesc(), "index", 1);
  
  const auto data3 = OP_CFG(DATA).ParentNodeIndex(0);
  DEF_GRAPH(sub_1) {
                     CHAIN(NODE("data3", data3)->NODE("netoutput3", NETOUTPUT));
                   };
  DEF_GRAPH(then_graph_) {
    CHAIN(NODE("data1", DATA)->NODE("mul1", MUL)->EDGE(0, 0)->NODE("netoutput1", NETOUTPUT));
    CHAIN(NODE("mul1", MUL)->EDGE(0, 0)->NODE("partitioned_call", PARTITIONEDCALL, sub_1)->NODE("netoutput1", NETOUTPUT));
  };
  auto then_graph = ToComputeGraph(then_graph_);

  auto partitioned_call_graph = ToComputeGraph(sub_1);
  then_graph_.Layout();
  auto partitioned_call_node = then_graph->FindNode("partitioned_call");
  partitioned_call_graph->SetParentNode(partitioned_call_node);
  main_graph->AddSubGraph(partitioned_call_graph);

  then_graph->SetName("then");
  ge::AttrUtils::SetInt(then_graph->FindFirstNodeMatchType(DATA)->GetOpDesc(), "index", 0);
  ge::AttrUtils::SetInt(then_graph->FindFirstNodeMatchType(DATA)->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, 1);
  ge::AttrUtils::SetInt(then_graph->FindFirstNodeMatchType(NETOUTPUT)->GetOpDesc(), "index", 0);
  ge::AttrUtils::SetInt(then_graph->FindFirstNodeMatchType(NETOUTPUT)->GetOpDesc()->MutableInputDesc(0), ge::ATTR_NAME_PARENT_NODE_INDEX, 0);

  auto else_graph = []() {
    DEF_GRAPH(g) {
      CHAIN(NODE("data2", DATA)->EDGE(0, 0)->NODE("mul2", MUL)->NODE("netoutput2", NETOUTPUT));
      CHAIN(NODE("data2", DATA)->EDGE(0, 0)->NODE("mul3", MUL)->NODE("netoutput2", NETOUTPUT));
    };
    return ToComputeGraph(g);
  }();
  else_graph->SetName("else");
  ge::AttrUtils::SetInt(else_graph->FindFirstNodeMatchType(DATA)->GetOpDesc(), "index", 0);
  ge::AttrUtils::SetInt(else_graph->FindFirstNodeMatchType(DATA)->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, 1);
  ge::AttrUtils::SetInt(else_graph->FindFirstNodeMatchType(NETOUTPUT)->GetOpDesc(), "index", 0);
  ge::AttrUtils::SetInt(else_graph->FindFirstNodeMatchType(NETOUTPUT)->GetOpDesc()->MutableInputDesc(0), ge::ATTR_NAME_PARENT_NODE_INDEX, 0);

  auto if_node = main_graph->FindFirstNodeMatchType("If");
  then_graph->SetParentGraph(main_graph);
  then_graph->SetParentNode(if_node);
  else_graph->SetParentGraph(main_graph);
  else_graph->SetParentNode(if_node);
  main_graph->AddSubgraph(then_graph);
  main_graph->AddSubgraph(else_graph);
  if_node->GetOpDesc()->AddSubgraphName("then");
  if_node->GetOpDesc()->AddSubgraphName("else");
  if_node->GetOpDesc()->SetSubgraphInstanceName(0, "then");
  if_node->GetOpDesc()->SetSubgraphInstanceName(1, "else");
  if_node->GetOpDesc()->AppendIrInput("cond", kIrInputRequired);
  if_node->GetOpDesc()->AppendIrInput("input", kIrInputDynamic);
  auto &names_indexes = if_node->GetOpDesc()->MutableAllInputName();
  names_indexes.clear();
  names_indexes["cast"] = 0;
  names_indexes["input"] = 1;
  auto net_output = main_graph->FindNode("netoutput");
  net_output->GetOpDesc()->SetSrcName({"if"});
  net_output->GetOpDesc()->SetSrcIndex({0});
  main_graph->TopologicalSorting();

  main_graph->SetGraphUnknownFlag(false);
  for (auto &sub_graph : main_graph->GetAllSubgraphs()) {
    sub_graph->SetGraphUnknownFlag(false);
  }
  AttrUtils::SetBool(main_graph, ATTR_NAME_NO_NEED_DYNAMIC_SHAPE_PARTITION, true);

  AddParentIndexForSubGraphNetoutput(main_graph);
  return main_graph;
}

/*
 *        input
 *          |
 *        cast0
 *          |
 *        while1
 *          |
 *        cast1
 *          |
 *       netoutput
 *
 * subgraph cond             subgraph body
 * +-------------------+     +-------------------+
 * | data0--netoutput0 |     | data1--netoutput1 |
 * +-------------------+     +-------------------+
 */
ComputeGraphPtr MemConflictShareGraph::BuildWhileDataToNetoutputGraph() {
  auto graph = []() {
    DEF_GRAPH(g) {
      CHAIN(NODE("input", "Data")->NODE("cast0", "Cast")->NODE("while1", "While")->NODE("cast1", "Cast")->NODE("netOutput", "NetOutput"));
    };
    return ToComputeGraph(g);
  }();
  graph->SetName("main");
  ge::AttrUtils::SetInt(graph->FindNode("cast0")->GetOpDesc(), "index", 0);

  AddWhileDummyCond(graph, "while1");
  AddWhileBody(graph, "while1");

  graph->TopologicalSorting();
  graph->SetGraphUnknownFlag(false);
  for (auto &sub_graph : graph->GetAllSubgraphs()) {
    sub_graph->SetGraphUnknownFlag(false);
  }
  AttrUtils::SetBool(graph, ATTR_NAME_NO_NEED_DYNAMIC_SHAPE_PARTITION, true);
  return graph;
}

/*
 *        input
 *          |
 *        cast0
 *          |
 *        while1
 *          |
 *        cast1
 *          |
 *       netoutput
 *
 * subgraph cond             subgraph body
 * +-------------------+     +-------------------------------+
 * | data0--netoutput0 |     | data1--while_mul1--netoutput1 |
 * +-------------------+     +-------------------------------+
 */
ComputeGraphPtr MemConflictShareGraph::BuildWhileDataMulNetoutputGraph() {
  auto graph = []() {
    DEF_GRAPH(g) {
      CHAIN(NODE("input", "Data")->NODE("cast0", "Cast")->NODE("while1", "While")->NODE("cast1", "Cast")->NODE("netOutput", "NetOutput"));
    };
    return ToComputeGraph(g);
  }();
  graph->SetName("main");
  ge::AttrUtils::SetInt(graph->FindNode("cast0")->GetOpDesc(), "index", 0);

  AddWhileDummyCond(graph, "while1");
  AddWhileBodyWithOneNode(graph, "while1");

  graph->TopologicalSorting();
  graph->SetGraphUnknownFlag(false);
  for (auto &sub_graph : graph->GetAllSubgraphs()) {
    sub_graph->SetGraphUnknownFlag(false);
  }
  AttrUtils::SetBool(graph, ATTR_NAME_NO_NEED_DYNAMIC_SHAPE_PARTITION, true);
  return graph;
}

/*
 *        input
 *          |
 *        cast0
 *          |
 *        while1
 *          |
 *        cast1
 *          |
 *       netoutput
 *
 * subgraph cond             subgraph body
 * +-------------------+     +-------------------------------------------+
 * | data0--netoutput0 |     | data1--while_mul1--while_mul2--netoutput1 |
 * +-------------------+     +-------------------------------------------+
 */
ComputeGraphPtr MemConflictShareGraph::BuildWhileDataMulMulNetoutputGraph() {
  auto graph = []() {
    DEF_GRAPH(g) {
      CHAIN(NODE("input", "Data")->NODE("cast0", "Cast")->NODE("while1", "While")->NODE("cast1", "Cast")->NODE("netOutput", "NetOutput"));
    };
    return ToComputeGraph(g);
  }();
  graph->SetName("main");
  ge::AttrUtils::SetInt(graph->FindNode("cast0")->GetOpDesc(), "index", 0);

  AddWhileDummyCond(graph, "while1");
  AddWhileBodyWithTwoNode(graph, "while1");

  graph->TopologicalSorting();
  graph->SetGraphUnknownFlag(false);
  for (auto &sub_graph : graph->GetAllSubgraphs()) {
    sub_graph->SetGraphUnknownFlag(false);
  }
  AttrUtils::SetBool(graph, ATTR_NAME_NO_NEED_DYNAMIC_SHAPE_PARTITION, true);
  return graph;
}

/*
 *    input1 input2
 *       |      |
 *     cast1  cast2
 *        \    /
 *        while1
 *         \ /
 *         add
 *          |
 *       netoutput
 *
 * subgraph cond             subgraph body
 * +-------------------+     +-------------------------+
 * | data0--netoutput0 |     | data1--\ /---netoutput1 |
 * +-------------------+     | data2--/ \--+           |
 *                           +-------------------------+
 */
ComputeGraphPtr MemConflictShareGraph::BuildWhileTwoDataToNetoutputExchangeGraph() {
  auto graph = []() {
    DEF_GRAPH(g) {
      CHAIN(NODE("input1", DATA)->NODE("cast1", CAST)->NODE("while1", WHILE)->NODE("add", ADD)->NODE("netOutput", NETOUTPUT));
      CHAIN(NODE("input2", DATA)->NODE("cast2", CAST)->NODE("while1", WHILE)->NODE("add", ADD));
    };
    return ToComputeGraph(g);
  }();
  graph->SetName("main");
  ge::AttrUtils::SetInt(graph->FindNode("cast1")->GetOpDesc(), "index", 0);
  ge::AttrUtils::SetInt(graph->FindNode("cast1")->GetOpDesc(), "index", 1);

  AddWhileDummyCond(graph, "while1");
  AddWhileBodyTwoExchange(graph, "while1");

  graph->TopologicalSorting();
  graph->SetGraphUnknownFlag(false);
  for (auto &sub_graph : graph->GetAllSubgraphs()) {
    sub_graph->SetGraphUnknownFlag(false);
  }
  AttrUtils::SetBool(graph, ATTR_NAME_NO_NEED_DYNAMIC_SHAPE_PARTITION, true);
  return graph;
}

/*
 *    input1 input2
 *       |      |
 *     cast1  cast2
 *        \    /
 *        while1
 *         \ /
 *         add
 *          |
 *       netoutput
 *
 * subgraph cond             subgraph body
 * +-------------------+     +----------------------+
 * | data0--netoutput0 |     | data1-----netoutput1 |
 * +-------------------+     | data2----+           |
 *                           +----------------------+
 */
ComputeGraphPtr MemConflictShareGraph::BuildWhileTwoDataToNetoutputNoExchangeGraph() {
  auto graph = []() {
    DEF_GRAPH(g) {
      CHAIN(NODE("input1", DATA)->NODE("cast1", CAST)->NODE("while1", WHILE)->NODE("add", ADD)->NODE("netOutput", NETOUTPUT));
      CHAIN(NODE("input2", DATA)->NODE("cast2", CAST)->NODE("while1", WHILE)->NODE("add", ADD));
    };
    return ToComputeGraph(g);
  }();
  graph->SetName("main");
  ge::AttrUtils::SetInt(graph->FindNode("cast1")->GetOpDesc(), "index", 0);
  ge::AttrUtils::SetInt(graph->FindNode("cast1")->GetOpDesc(), "index", 1);

  AddWhileDummyCond(graph, "while1");
  AddWhileBodyTwoNoExchange(graph, "while1");

  graph->TopologicalSorting();
  graph->SetGraphUnknownFlag(false);
  for (auto &sub_graph : graph->GetAllSubgraphs()) {
    sub_graph->SetGraphUnknownFlag(false);
  }
  AttrUtils::SetBool(graph, ATTR_NAME_NO_NEED_DYNAMIC_SHAPE_PARTITION, true);
  return graph;
}

void AddWhileBodyWithRef(ge::ComputeGraphPtr parent_graph, const std::string &parent_node_name) {
  auto body_graph = []() {
    const auto refnode = OP_CFG(ASSIGNADD).Attr(ATTR_NAME_REFERENCE, true).InNames({"ref"}).OutNames({"ref"});
    DEF_GRAPH(g) {
      CHAIN(NODE("data1", DATA)->NODE("refnode", refnode)->NODE("netoutput1", NETOUTPUT));
    };
    return ToComputeGraph(g);
  }();
  body_graph->SetName("body");

  ge::AttrUtils::SetInt(body_graph->FindFirstNodeMatchType("Data")->GetOpDesc(), "index", 0);
  ge::AttrUtils::SetInt(body_graph->FindFirstNodeMatchType("Data")->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, 0);
  ge::AttrUtils::SetInt(body_graph->FindFirstNodeMatchType("NetOutput")->GetOpDesc(), "index", 0);
  ge::AttrUtils::SetInt(body_graph->FindFirstNodeMatchType("NetOutput")->GetOpDesc()->MutableInputDesc(0), ge::ATTR_NAME_PARENT_NODE_INDEX, 0);

  AddSubgraphInstance(parent_graph, body_graph, 1, parent_node_name);
}

/*
 *        input
 *          |
 *        cast0
 *          |
 *        while1
 *          |
 *        cast1
 *          |
 *       netoutput
 *
 * subgraph cond             subgraph body
 * +-------------------+     +----------------------------+
 * | data0--netoutput0 |     | data1--refnode--netoutput1 |
 * +-------------------+     +----------------------------+
 */
ComputeGraphPtr MemConflictShareGraph::BuildWhileDataRefNetoutputGraph() {
  auto graph = []() {
    DEF_GRAPH(g) {
      CHAIN(NODE("input", "Data")->NODE("cast0", "Cast")->NODE("while1", "While")->NODE("cast1", "Cast")->NODE("netOutput", "NetOutput"));
    };
    return ToComputeGraph(g);
  }();
  graph->SetName("main");
  ge::AttrUtils::SetInt(graph->FindNode("cast0")->GetOpDesc(), "index", 0);

  AddWhileDummyCond(graph, "while1");
  AddWhileBodyWithRef(graph, "while1");

  graph->TopologicalSorting();
  graph->SetGraphUnknownFlag(false);
  for (auto &sub_graph : graph->GetAllSubgraphs()) {
    sub_graph->SetGraphUnknownFlag(false);
  }
  AttrUtils::SetBool(graph, ATTR_NAME_NO_NEED_DYNAMIC_SHAPE_PARTITION, true);
  return graph;
}

void AddWhileBodyWithPartionedCall(ge::ComputeGraphPtr &parent_graph, const std::string &parent_node_name) {
  const auto data2 = OP_CFG(DATA).ParentNodeIndex(0);
  DEF_GRAPH(sub_1) {
                     CHAIN(NODE("data2", data2)->NODE("netoutput2", NETOUTPUT));
                   };
  const auto data1 = OP_CFG(DATA).ParentNodeIndex(1);
  DEF_GRAPH(body) {
                    CHAIN(NODE("data1", data1)->NODE("partitioned_call", PARTITIONEDCALL, sub_1)->NODE("netoutput1", NETOUTPUT));
                  };

  auto body_graph = ToComputeGraph(body);
  body_graph->SetName("body");

  ge::AttrUtils::SetInt(body_graph->FindFirstNodeMatchType("Data")->GetOpDesc(), "index", 0);
  ge::AttrUtils::SetInt(body_graph->FindFirstNodeMatchType("Data")->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, 0);
  ge::AttrUtils::SetInt(body_graph->FindFirstNodeMatchType("NetOutput")->GetOpDesc(), "index", 0);
  ge::AttrUtils::SetInt(body_graph->FindFirstNodeMatchType("NetOutput")->GetOpDesc()->MutableInputDesc(0), ge::ATTR_NAME_PARENT_NODE_INDEX, 0);

  auto partitioned_call_graph = ToComputeGraph(sub_1);
  body.Layout();
  auto partitioned_call_node = body_graph->FindNode("partitioned_call");
  partitioned_call_graph->SetParentNode(partitioned_call_node);
  parent_graph->AddSubGraph(partitioned_call_graph);

  AddSubgraphInstance(parent_graph, body_graph, 1, parent_node_name);

  AddParentIndexForSubGraphNetoutput(parent_graph);
}

/*
 *        input
 *          |
 *        cast0
 *          |
 *        while1
 *          |
 *        cast1
 *          |
 *       netoutput
 *
 * subgraph cond       subgraph body
 * +-------------+     +------------------------------------+
 * |   data0     |     |     data1                          |
 * |     |       |     |       |                            |
 * | netoutput0  |     |  partitioned_call   +------------+ |
 * +-------------+     |       |             |   data2    | |
 *                     |   netoutput1        |     |      | |
 *                     |                     | netoutput2 | |
 *                     |                     +------------+ |
 *                     +------------------------------------+                      
 *
 */
ComputeGraphPtr MemConflictShareGraph::BuildWhileDataPartitionedCallNetoutputGraph() {
  auto graph = []() {
    DEF_GRAPH(g) {
      CHAIN(NODE("input", "Data")->NODE("cast0", "Cast")->NODE("while1", "While")->NODE("cast1", "Cast")->NODE("netOutput", "NetOutput"));
    };
    return ToComputeGraph(g);
  }();
  graph->SetName("main");
  ge::AttrUtils::SetInt(graph->FindNode("cast0")->GetOpDesc(), "index", 0);

  AddWhileDummyCond(graph, "while1");
  AddWhileBodyWithPartionedCall(graph, "while1");

  graph->TopologicalSorting();
  graph->SetGraphUnknownFlag(false);
  for (auto &sub_graph : graph->GetAllSubgraphs()) {
    sub_graph->SetGraphUnknownFlag(false);
  }
  AttrUtils::SetBool(graph, ATTR_NAME_NO_NEED_DYNAMIC_SHAPE_PARTITION, true);
  return graph;
}

/*
 *        input
 *          |
 *        cast0
 *          |\
 *          |  \
 *        while1 relu
 *          |     |
 *        cast1   |
 *          |    /
 *        netoutput
 *
 * subgraph cond             subgraph body
 * +-------------------+     +-------------------+
 * | data0--netoutput0 |     | data1--netoutput1 |
 * +-------------------+     +-------------------+
 */
ComputeGraphPtr MemConflictShareGraph::BuildWhileInNodeConnectToMultiNodesGraph() {
  auto graph = []() {
    DEF_GRAPH(g) {
                   CHAIN(NODE("input", "Data")->NODE("cast0", "Cast")->NODE("while1", "While")->NODE("cast1", "Cast")->NODE("netOutput", "NetOutput"));
                   CHAIN(NODE("cast0", "Cast")->EDGE(0, 0)->NODE("relu", RELU)->NODE("netOutput", "NetOutput"));
                 };
    return ToComputeGraph(g);
  }();
  graph->SetName("main");
  ge::AttrUtils::SetInt(graph->FindNode("cast0")->GetOpDesc(), "index", 0);

  AddWhileDummyCond(graph, "while1");
  AddWhileBody(graph, "while1");

  graph->TopologicalSorting();
  graph->SetGraphUnknownFlag(false);
  for (auto &sub_graph : graph->GetAllSubgraphs()) {
    sub_graph->SetGraphUnknownFlag(false);
  }
  AttrUtils::SetBool(graph, ATTR_NAME_NO_NEED_DYNAMIC_SHAPE_PARTITION, true);
  return graph;
}

/*
 * data1  data2
 *   |      |
 *   a      b
 *   \      /
 *   phony_concat (NoPaddingContinuousInput,
 *      |           and otuput ref input)
 *    netoutput
 */
ComputeGraphPtr MemConflictShareGraph::BuildUnknowShapeGraph() {
  DEF_GRAPH(g1) {
                  CHAIN(NODE("data1", DATA)->NODE("a", RELU)->NODE("phony_concat", PHONYCONCAT)->NODE("netoutput", NETOUTPUT));
                  CHAIN(NODE("data2", DATA)->NODE("b", RELU)->NODE("phony_concat", PHONYCONCAT));
                };
  auto graph = ToComputeGraph(g1);
  SetNoPaddingContinuousInput(graph, "phony_concat");
  auto netoutput_node = graph->FindNode("netoutput");
  if (netoutput_node != nullptr) {
    auto tensor_desc = netoutput_node->GetOpDescBarePtr()->MutableInputDesc(0);
    auto shape = GeShape({-1, 2, 2, 2});
    tensor_desc->SetShape(shape);
  }
  return graph;
}

/*
 *        split1
 *         /  \
 *     split2  c
 *     /   \
 *    a     b
 */
ComputeGraphPtr MemConflictShareGraph::BuildNotContinuousOutGraph() {
  DEF_GRAPH(g1) {
                  CHAIN(NODE("split1", SPLIT)->NODE("split2", SPLIT));
                  CHAIN(NODE("split1", SPLIT)->NODE("c", RELU)->NODE("netoutput", NETOUTPUT));
                  CHAIN(NODE("split2", SPLIT)->NODE("a", CAST)->NODE("netoutput", NETOUTPUT));
                  CHAIN(NODE("split2", SPLIT)->NODE("b", RELU)->NODE("netoutput", NETOUTPUT));
                };
  auto graph = ToComputeGraph(g1);

  auto a = graph->FindNode("a");
  a->GetOpDesc()->MutableInputDesc(0)->SetDataType(DT_FLOAT16);
  a->GetOpDesc()->MutableInputDesc(0)->SetOriginDataType(DT_FLOAT16);
  a->GetOpDesc()->MutableOutputDesc(0)->SetDataType(DT_INT32);
  a->GetOpDesc()->MutableOutputDesc(0)->SetOriginDataType(DT_INT32);
  // 如果不设置，会被设置为动态shape图
  AttrUtils::SetBool(graph, ATTR_NAME_NO_NEED_DYNAMIC_SHAPE_PARTITION, true);
  return graph;
}

/*
┌────────┐  (1,0)   ┌───┐  (0,0)   ┌────────┐  (0,0)   ┌───────────┐
│ split1 │ ───────> │ c │ ───────> │ while1 │ ───────> │ netoutput │ <┐
└────────┘          └───┘          └────────┘          └───────────┘  │
  │                                                      ∧            │
  │ (0,0)                                                │            │
  ∨                                                      │            │
┌────────┐  (0,0)   ┌───┐  (0,1)                         │            │
│ split2 │ ───────> │ a │ ───────────────────────────────┘            │ (0,2)
└────────┘          └───┘                                             │
  │                                                                   │
  │ (1,0)                                                             │
  ∨                                                                   │
┌────────┐                                                            │
│   b    │ ───────────────────────────────────────────────────────────┘
└────────┘

               g

┌───────┐  (0,0)   ┌────────────┐
│ data0 │ ───────> │ netoutput0 │
└───────┘          └────────────┘

                           g

┌───────┐  (0,0)   ┌────────────┐  (0,0)   ┌────────────┐
│ data1 │ ───────> │ while_mul1 │ ───────> │ netoutput1 │
└───────┘          └────────────┘          └────────────┘
 */
ComputeGraphPtr MemConflictShareGraph::BuildWhileSplitGraph() {
  auto graph = []() {
    DEF_GRAPH(g) {
                CHAIN(NODE("split1", SPLIT)->NODE("split2", SPLIT));
                CHAIN(NODE("split1", SPLIT)->NODE("c", RELU)->NODE("while1", "While")->NODE("netoutput", NETOUTPUT));
                CHAIN(NODE("split2", SPLIT)->NODE("acast", CAST)->NODE("netoutput", NETOUTPUT));
                CHAIN(NODE("split2", SPLIT)->NODE("b", RELU)->NODE("netoutput", NETOUTPUT));
              };
    return ToComputeGraph(g);
  }();
  graph->SetName("main");
  ge::AttrUtils::SetInt(graph->FindNode("c")->GetOpDesc(), "index", 0);
  auto cast_node = graph->FindNode("acast");
  cast_node->GetOpDesc()->MutableInputDesc(0)->SetDataType(DT_FLOAT16);
  cast_node->GetOpDesc()->MutableInputDesc(0)->SetOriginDataType(DT_FLOAT16);
  cast_node->GetOpDesc()->MutableOutputDesc(0)->SetDataType(DT_INT32);
  cast_node->GetOpDesc()->MutableOutputDesc(0)->SetOriginDataType(DT_INT32);
  AttrUtils::SetListListInt(cast_node->GetOpDesc(), ATTR_NAME_OUTPUT_INPLACE_ABILITY, {{0, 0}});

  AddWhileDummyCond(graph, "while1");
  AddWhileBodyWithOneNode(graph, "while1");

  graph->TopologicalSorting();
  graph->SetGraphUnknownFlag(false);
  for (auto &sub_graph : graph->GetAllSubgraphs()) {
    sub_graph->SetGraphUnknownFlag(false);
  }
  AttrUtils::SetBool(graph, ATTR_NAME_NO_NEED_DYNAMIC_SHAPE_PARTITION, true);
  return graph;
}
} // namespace ge
