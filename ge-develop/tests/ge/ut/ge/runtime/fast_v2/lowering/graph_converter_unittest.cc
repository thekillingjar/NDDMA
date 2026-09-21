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
#include "lowering/graph_converter.h"
#include "exe_graph/lowering/exe_graph_attrs.h"
#include "exe_graph/lowering/frame_selector.h"
#include "graph/utils/node_utils.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "graph/compute_graph.h"
#include "graph/utils/graph_utils.h"
#include "framework/common/ge_types.h"
#include "framework/runtime/gert_const_types.h"
#include "common/share_graph.h"
#include "faker/global_data_faker.h"
#include "faker/aicpu_taskdef_faker.h"
#include "kernel/memory/caching_mem_allocator.h"
#include <queue>
#include "common/bg_test.h"
#include "common/table_driven.h"
#include "exe_graph/runtime/continuous_buffer.h"
#include "exe_graph/runtime/compute_node_info.h"
#include "lowering/placement/placed_lowering_result.h"
#include "exe_graph/lowering/lowering_definitions.h"
#include "faker/model_desc_holder_faker.h"
#include "common/summary_checker.h"
#include "common/topo_checker.h"
#include "kernel/common_kernel_impl/memory_copy.h"
#include "kernel/common_kernel_impl/copy_flow_launch.h"
#include "graph/utils/graph_dump_utils.h"

using namespace ge;
namespace gert {
extern const LowerResult *LoweringComputeGraph(const ge::ComputeGraphPtr &graph, LoweringGlobalData &global_data);
namespace {
constexpr const char *kAtomicLaunchKernelType = "AtomicLaunchKernelWithFlag";
void CheckAtomicInfoInComputeNodeInfo(const ge::ExecuteGraph *const exe_graph,
                                      const ge::FastNode *const atomic_kernel_node,
                                      const char *atomic_compute_node_name, size_t atomic_compute_node_input_num) {
  Buffer bf;
  ASSERT_EQ(ge::AttrUtils::GetZeroCopyBytes(exe_graph, kComputeNodeInfo, bf), true);
  auto c_nodes_info = reinterpret_cast<ContinuousBuffer *>(bf.GetData());

  size_t atomic_compute_node_info_idx = 0U;
  ge::AttrUtils::GetInt(atomic_kernel_node->GetOpDescBarePtr(), kComputeNodeIndex, atomic_compute_node_info_idx);
  auto compute_node_info = c_nodes_info->Get<ComputeNodeInfo>(atomic_compute_node_info_idx);
  EXPECT_EQ(compute_node_info->GetInputsNum(), atomic_compute_node_input_num);

  auto atomic_node_name_index = reinterpret_cast<size_t>(compute_node_info->GetNodeName());
  ASSERT_EQ(ge::AttrUtils::GetZeroCopyBytes(exe_graph, kBuffer, bf), true);
  auto buffer = reinterpret_cast<ContinuousBuffer *>(bf.GetData());
  auto atomic_node_name = buffer->Get<char>(atomic_node_name_index);
  EXPECT_STREQ(atomic_node_name, atomic_compute_node_name);
}
bool IsReleaseNode(const FastNode *const release_node) {
  int32_t release_index;
  return AttrUtils::GetInt(release_node->GetOpDescBarePtr(), kReleaseResourceIndex, release_index);
}
ge::EdgeSrcEndpoint GetPeer(const ge::FastNode *inner_data) {
  int32_t parent_index;
  while (true) {
    if (!AttrUtils::GetInt(inner_data->GetOpDescBarePtr(), "index", parent_index)) {
      throw std::invalid_argument("The InnerData node is invalid, which does not have a index attr");
    }
    auto parent_node = inner_data->GetExtendInfo()->GetOwnerGraphBarePtr()->GetParentNodeBarePtr();
    if (parent_node == nullptr) {
      throw std::invalid_argument("The InnerData node is invalid, which does not have a parent node");
    }
    const auto in_data_edge = parent_node->GetInDataEdgeByIndex(parent_index);
    if (in_data_edge == nullptr) {
      throw std::invalid_argument("The InnerData node has no peer or the parent index is invalid");
    }
    auto src_endpoint = ge::FastNodeUtils::GetSrcEndpoint(in_data_edge);
    auto src_node = src_endpoint.node;
    if (src_node->GetType() != "InnerData") {
      return src_endpoint;
    }
    inner_data = src_node;
  }
}

bool IsRootGraph(const ExecuteGraph *const graph) {
  return graph->GetParentGraphBarePtr() == nullptr;
}

bool IsAllocNodeInTheSameGraph(const ge::FastNode *const release_node) {
  int32_t release_index;
  if (!AttrUtils::GetInt(release_node->GetOpDescBarePtr(), kReleaseResourceIndex, release_index)) {
    throw std::invalid_argument("the node is not a release node");
  }
  auto in_node = ge::FastNodeUtils::GetInDataNodeByIndex(release_node, release_index);
  if (in_node->GetType() != "InnerData") {
    return true;
  }
  auto src_endpoint = GetPeer(in_node);
  return !IsRootGraph(src_endpoint.node->GetExtendInfo()->GetOwnerGraphBarePtr());
}
bool IsAllocNodeInInitGraph(const ge::FastNode *const release_node) {
  int32_t release_index;
  if (!AttrUtils::GetInt(release_node->GetOpDescBarePtr(), kReleaseResourceIndex, release_index)) {
    throw std::invalid_argument("the node is not a release node");
  }
  auto inner_data = ge::FastNodeUtils::GetInDataNodeByIndex(release_node, release_index);
  if (inner_data->GetType() != "InnerData") {
    throw std::invalid_argument("The same graph, should not be here(call IsAllocNodeInTheSameGraph first)");
  }
  auto src_endpoint = GetPeer(inner_data);
  return src_endpoint.node->GetType() == "Init";
}

LowerResult LoweringAssign(const ge::NodePtr &node, const LowerInput &lower_input) {
  (void)node;
  (void)lower_input;
  return {HyperStatus::Success(), {}, {}, {}};
}
REGISTER_NODE_CONVERTER("_lower_assign", LoweringAssign);
}  // namespace
class GraphConverterUT : public bg::BgTest {
 public:
  void CheckAllocFreeStageMatch(ExecuteGraph *const graph) {
    auto init_graph =
        ge::FastNodeUtils::GetSubgraphFromNode(ge::ExecuteGraphUtils::FindFirstNodeMatchType(graph, "Init"), 0);
    auto main_graph =
        ge::FastNodeUtils::GetSubgraphFromNode(ge::ExecuteGraphUtils::FindFirstNodeMatchType(graph, "Main"), 0);
    auto de_init_graph =
        ge::FastNodeUtils::GetSubgraphFromNode(ge::ExecuteGraphUtils::FindFirstNodeMatchType(graph, "DeInit"), 0);
    ASSERT_NE(init_graph, nullptr);
    ASSERT_NE(main_graph, nullptr);
    ASSERT_NE(de_init_graph, nullptr);

    for (const auto &node : main_graph->GetAllNodes()) {
      if (!IsReleaseNode(node)) {
        continue;
      }
      // main图上的释放资源的节点，必然释放本图的资源（因为main图前面只有init图），
      // 如果main图上的节点释放了非本图的节点，那必然是释放了init图上的节点，
      // 由于init图只执行一遍，main图可以执行多遍，这里就出现了一次申请，多次释放，出错了
      EXPECT_TRUE(IsAllocNodeInTheSameGraph(node))
          << "The release node " << node->GetName() << ", type " << node->GetType()
          << " in MainGraph, release a node does not in this graph";
    }
    for (const auto &node : de_init_graph->GetAllNodes()) {
      if (!IsReleaseNode(node)) {
        continue;
      }
      if (IsAllocNodeInTheSameGraph(node)) {
        continue;
      }
      EXPECT_TRUE(IsAllocNodeInInitGraph(node)) << "The release node " << node->GetName() << ", type "
                                                << node->GetType() << " in DeInitGraph, release a node in MainGraph";
    }
  }
};
namespace {
template <typename Nodes>
std::set<std::string> GetNodeTypes(const Nodes &nodes) {
  std::set<std::string> node_types;
  for (const auto &node : nodes) {
    node_types.emplace(node->GetType());
  }
  return node_types;
}
}  // namespace

TEST_F(GraphConverterUT, OrderedHolders_ConnectOk) {
  auto graph = ShareGraph::BuildNoOpGraph();
  ASSERT_NE(graph, nullptr);
  graph->TopologicalSorting();
  SetGraphOutShapeRange(graph);
auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).FakeWithHandleAiCore("Add", false, false).Build();
  auto model_desc_holder = ModelDescHolderFaker().Build();
  auto exe_graph =
      GraphConverter().SetModelDescHolder(&model_desc_holder).ConvertComputeGraphToExecuteGraph(graph, global_data);
  ASSERT_NE(exe_graph, nullptr);

  const std::string kernel_type = "LaunchKernelWithHandle";
  const std::string add1_kernel_launch_node_name_prefix = kernel_type + "_" + "add1";
  const std::string add2_kernel_launch_node_name_prefix = kernel_type + "_" + "add2";
  ge::FastNode *add1_kernel_launch_node = nullptr;
  ge::FastNode *add2_kernel_launch_node = nullptr;
  auto main_exe_graph =
      ge::FastNodeUtils::GetSubgraphFromNode(ge::ExecuteGraphUtils::FindFirstNodeMatchType(exe_graph.get(), "Main"), 0);
  for (auto &node : main_exe_graph->GetDirectNode()) {
    if (node == nullptr) {
      continue;
    }
    if (node->GetType() == kernel_type) {
      string node_name = node->GetName();
      if (node_name.substr(0, add1_kernel_launch_node_name_prefix.length()) == add1_kernel_launch_node_name_prefix) {
        add1_kernel_launch_node = node;
      }
      if (node_name.substr(0, add2_kernel_launch_node_name_prefix.length()) == add2_kernel_launch_node_name_prefix) {
        add2_kernel_launch_node = node;
      }
    }
  }
  ASSERT_NE(add1_kernel_launch_node, nullptr);
  ASSERT_NE(add2_kernel_launch_node, nullptr);
  EXPECT_EQ(add1_kernel_launch_node->IsDirectlyControlledByNode(add2_kernel_launch_node), true);
}

TEST_F(GraphConverterUT, MainGraphWithoutAnyOutputs_ConstructOk) {
  auto graph = ShareGraph::SimpleVariableAssignGraph();
  ASSERT_NE(graph, nullptr);
  auto assign = graph->FindFirstNodeMatchType("Assign");
  ge::AttrUtils::SetStr(assign->GetOpDesc(), "_ge_attr_lowering_func", "_lower_assign");
  graph->TopologicalSorting();

  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).Build();
  auto model_desc_holder = ModelDescHolderFaker().Build();
  auto exe_graph =
      GraphConverter().SetModelDescHolder(&model_desc_holder).ConvertComputeGraphToExecuteGraph(graph, global_data);
  ASSERT_NE(exe_graph, nullptr);

  auto main_exe_graph =
      ge::FastNodeUtils::GetSubgraphFromNode(ge::ExecuteGraphUtils::FindFirstNodeMatchType(exe_graph.get(), "Main"), 0);
  auto ensure_tensor_at_out_mem =
      ge::ExecuteGraphUtils::FindFirstNodeMatchType(main_exe_graph, "EnsureTensorAtOutMemory");
  ASSERT_EQ(ensure_tensor_at_out_mem, nullptr);  // netoutput没有数据输入，也不会lowering出EnsureTensorAtOutMemory
}

TEST_F(GraphConverterUT, OrderedHoldersConnectOk_BothFromInitNode) {
  std::string file_name_0 = "test_weight_0.bin";
  std::string file_name_1 = "test_weight_1.bin";
  int32_t data[25];
  for (int32_t i = 0; i < 25; i++) {
    data[i] = i;
  }
  std::ofstream out0(file_name_0, std::ios::binary);
  if (!out0.is_open()) {
    return;
  }
  out0.write(reinterpret_cast<char *>(data), sizeof(data));
  out0.close();

  std::ofstream out1(file_name_1, std::ios::binary);
  if (!out1.is_open()) {
    return;
  }
  out1.write(reinterpret_cast<char *>(data), sizeof(data));
  out1.close();

  auto graph = ShareGraph::Build2FileConstantWithCtrlEdgeGraph();

  auto FileConstant0 = graph->FindNode("FileConstant0");
  ge::AttrUtils::SetStr(FileConstant0->GetOpDesc(), "location", file_name_0);
  auto FileConstant1 = graph->FindNode("FileConstant1");
  ge::AttrUtils::SetStr(FileConstant1->GetOpDesc(), "location", file_name_1);
  std::vector<int32_t> shape = {5, 5};
  ge::AttrUtils::SetListInt(FileConstant0->GetOpDesc(), "shape", shape);
  ge::AttrUtils::SetListInt(FileConstant1->GetOpDesc(), "shape", shape);
  graph->TopologicalSorting();

  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).Build();
  ModelDescHolder model_desc_holder = ModelDescHolderFaker().Build();
  model_desc_holder.SetSpaceRegistry(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry());
  auto graph_convert = GraphConverter().SetModelDescHolder(&model_desc_holder);
  auto exe_graph = graph_convert.ConvertComputeGraphToExecuteGraph(graph, global_data);
  ASSERT_NE(exe_graph, nullptr);

  const std::string kernel_type = "FileConstantKernel";
  const std::string fileconstant0_kernel_name_prefix = kernel_type + "_" + "FileConstant0";
  const std::string fileconstant1_kernel_name_prefix = kernel_type + "_" + "FileConstant1";
  ge::FastNode *fileconstant0_kernel = nullptr;
  ge::FastNode *fileconstant1_kernel = nullptr;
  auto main_exe_graph =
      ge::FastNodeUtils::GetSubgraphFromNode(ge::ExecuteGraphUtils::FindFirstNodeMatchType(exe_graph.get(), "Init"), 0);
  for (auto &node : main_exe_graph->GetDirectNode()) {
    if (node == nullptr) {
      continue;
    }
    if (node->GetType() == kernel_type) {
      string node_name = node->GetName();
      if (node_name.substr(0, fileconstant0_kernel_name_prefix.length()) == fileconstant0_kernel_name_prefix) {
        fileconstant0_kernel = node;
      }
      if (node_name.substr(0, fileconstant1_kernel_name_prefix.length()) == fileconstant1_kernel_name_prefix) {
        fileconstant1_kernel = node;
      }
    }
  }
  ASSERT_NE(fileconstant0_kernel, nullptr);
  ASSERT_NE(fileconstant1_kernel, nullptr);
  EXPECT_EQ(fileconstant1_kernel->IsDirectlyControlledByNode(fileconstant0_kernel), true);
}

// TEST_F(GraphConverterUT, OrderedHoldersConnectOk_MeetNoOrderedHoldersNode) {
//   // todo
// }

TEST_F(GraphConverterUT, LoweringSingleNodeGraph) {
  auto graph = ShareGraph::BuildSingleNodeGraph();
  graph->TopologicalSorting();
auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).FakeWithHandleAiCore("Add", false).Build();
  ASSERT_NE(graph, nullptr);
  auto model_desc_holder = ModelDescHolderFaker().Build();
  auto exe_graph =
      GraphConverter().SetModelDescHolder(&model_desc_holder).ConvertComputeGraphToExecuteGraph(graph, global_data);
  ASSERT_NE(exe_graph, nullptr);
  ge::DumpGraph(exe_graph.get(), "LoweringSingleNodeGraph");

  CheckGraphGenerally(exe_graph.get());
  EXPECT_EQ(GetNodeTypes(exe_graph->GetDirectNode()), std::set<std::string>({"Init", "Main", "DeInit"}));
  CheckAllocFreeStageMatch(exe_graph.get());
  // todo check more
}

TEST_F(GraphConverterUT, LoweringSingleNodeGraph_NoDataDependent_Ok) {
  auto graph = ShareGraph::BuildSingleNodeGraph();
  auto reshape = graph->FindNode("add1");

  graph->TopologicalSorting();
auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).FakeWithHandleAiCore("Add", false).Build();
  ASSERT_NE(graph, nullptr);
  auto model_desc_holder = ModelDescHolderFaker().Build();
  auto exe_graph =
      GraphConverter().SetModelDescHolder(&model_desc_holder).ConvertComputeGraphToExecuteGraph(graph, global_data);
  ASSERT_NE(exe_graph, nullptr);

  auto nodes = GetNodeTypes(exe_graph->GetAllNodes());
  ASSERT_EQ(nodes.count("MakeSureTensorAtHost"), 0);
}

TEST_F(GraphConverterUT, LoweringSingleNodeGraph_DataDependent_OK) {
  auto graph = ShareGraph::BuildSingleNodeGraph("Reshape");
  auto reshape = graph->FindNode("add1");
  reshape->GetOpDesc()->AppendIrInput("x", kIrInputRequired);
  reshape->GetOpDesc()->AppendIrInput("shape", kIrInputRequired);

  graph->TopologicalSorting();
auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).FakeWithHandleAiCore("Reshape", false).Build();
  ASSERT_NE(graph, nullptr);
  auto model_desc_holder = ModelDescHolderFaker().Build();
  auto exe_graph =
      GraphConverter().SetModelDescHolder(&model_desc_holder).ConvertComputeGraphToExecuteGraph(graph, global_data);
  ASSERT_NE(exe_graph, nullptr);

  CheckGraphGenerally(exe_graph.get());
  CheckAllocFreeStageMatch(exe_graph.get());
  auto nodes = GetNodeTypes(exe_graph->GetAllNodes());
  ASSERT_EQ(nodes.count("MakeSureTensorAtHost"), 1);
}

TEST_F(GraphConverterUT, LoweringSingleNodeGraph_DataDependent_LessIrInputs_Failed) {
  auto graph = ShareGraph::BuildSingleNodeGraph("Reshape");
  auto reshape = graph->FindNode("add1");

  graph->TopologicalSorting();
auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).FakeWithHandleAiCore("Reshape", false).Build();
  ASSERT_NE(graph, nullptr);
  auto model_desc_holder = ModelDescHolderFaker().Build();
  auto exe_graph =
      GraphConverter().SetModelDescHolder(&model_desc_holder).ConvertComputeGraphToExecuteGraph(graph, global_data);
  ASSERT_EQ(exe_graph, nullptr);
}

TEST_F(GraphConverterUT, LoweringSingleNodeGraph_FailedThenSuccess) {
  auto graph = ShareGraph::BuildSingleNodeGraph("Reshape");
  auto reshape = graph->FindNode("add1");

  graph->TopologicalSorting();
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).FakeWithHandleAiCore("Reshape", false).Build();
  ASSERT_NE(graph, nullptr);
  auto model_desc_holder = ModelDescHolderFaker().Build();
  auto exe_graph =
      GraphConverter().SetModelDescHolder(&model_desc_holder).ConvertComputeGraphToExecuteGraph(graph, global_data);
  ASSERT_EQ(exe_graph, nullptr);

  global_data = GlobalDataFaker(root_model).FakeWithHandleAiCore("Reshape", false).Build();
  reshape->GetOpDesc()->AppendIrInput("x", kIrInputRequired);
  reshape->GetOpDesc()->AppendIrInput("shape", kIrInputRequired);
  auto exe_graph1 =
      GraphConverter().SetModelDescHolder(&model_desc_holder).ConvertComputeGraphToExecuteGraph(graph, global_data);
  ASSERT_NE(exe_graph1, nullptr);
}


TEST_F(GraphConverterUT, LoweringLstmppGraphOk) {
  auto graph = ShareGraph::LstmpGraph();
  graph->TopologicalSorting();
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model)
                         .FakeWithoutHandleAiCore("TransData", true)
                         .FakeWithoutHandleAiCore("DynamicRNNV3", false)
                         .Build();

  ASSERT_NE(graph, nullptr);
  graph->TopologicalSorting();
  GraphUtils::DumpGEGraphToOnnx(*graph, "LoweringLstmppGraphCompute");

  auto model_desc_holder = ModelDescHolderFaker().Build();
  auto exe_graph =
      GraphConverter().SetModelDescHolder(&model_desc_holder).ConvertComputeGraphToExecuteGraph(graph, global_data);
  ASSERT_NE(exe_graph, nullptr);

  CheckGraphGenerally(exe_graph.get());
  EXPECT_EQ(GetNodeTypes(exe_graph->GetDirectNode()), std::set<std::string>({"Init", "Main", "DeInit"}));
  ge::DumpGraph(exe_graph.get(), "LoweringLstmppGraphExe");
  CheckAllocFreeStageMatch(exe_graph.get());
  // todo check
  exe_graph->TopologicalSorting();
  memory::CachingMemAllocator::GetAllocator()->Finalize();
}

TEST_F(GraphConverterUT, LoweringShapeReshapePattern) {
  auto graph = ShareGraph::BuildShapeToReshapeGraph();
  graph->TopologicalSorting();
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model)
                         .FakeWithoutHandleAiCore("Concat", false)
                         .FakeWithoutHandleAiCore("Gather", false)
                         .Build();

  ASSERT_NE(graph, nullptr);
  graph->TopologicalSorting();
  GraphUtils::DumpGEGraphToOnnx(*graph, "ComputeGraphShapeReshape");
  graph->SetGraphUnknownFlag(true);
  auto model_desc_holder = ModelDescHolderFaker().Build();
  auto exe_graph =
      GraphConverter().SetModelDescHolder(&model_desc_holder).ConvertComputeGraphToExecuteGraph(graph, global_data);
  ASSERT_NE(exe_graph, nullptr);

  CheckGraphGenerally(exe_graph.get());
  EXPECT_EQ(GetNodeTypes(exe_graph->GetDirectNode()), std::set<std::string>({"Init", "Main", "DeInit"}));
  ge::DumpGraph(exe_graph.get(), "ExeGraphShapeReshape");
  CheckAllocFreeStageMatch(exe_graph.get());

  // todo check
  exe_graph->TopologicalSorting();
  memory::CachingMemAllocator::GetAllocator()->Finalize();
}

TEST_F(GraphConverterUT, LoweringClearStepContainerGraph) {
  auto graph = ShareGraph::BuildHostCpuDataFlowGraph();
  graph->FindNode("add1")->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCore);
  graph->TopologicalSorting();
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).FakeWithHandleAiCore("Add", false).Build();
  ASSERT_NE(graph, nullptr);

  bg::ValueHolder::PushGraphFrame();
  auto clear_builder = [&]() -> bg::ValueHolderPtr {
    return bg::ValueHolder::CreateVoid<bg::ValueHolder>("ClearStepContainer", {});
  };
  auto clear_holder = bg::FrameSelector::OnMainRootLast(clear_builder);
  EXPECT_NE(clear_holder, nullptr);
  auto last_exec_nodes = bg::ValueHolder::GetLastExecNodes();
  EXPECT_EQ(last_exec_nodes.size(), 1);
  EXPECT_NE(last_exec_nodes[0], nullptr);
  std::string clear_resource_name = last_exec_nodes[0]->GetFastNode()->GetOpDescBarePtr()->GetName();
  EXPECT_EQ(clear_resource_name.find("ClearStepContainer"), 0);
  bg::ValueHolder::PopGraphFrame();

  auto model_desc_holder = ModelDescHolderFaker().Build();
  auto exe_graph =
      GraphConverter().SetModelDescHolder(&model_desc_holder).ConvertComputeGraphToExecuteGraph(graph, global_data);
  ASSERT_NE(exe_graph, nullptr);
  ge::DumpGraph(exe_graph.get(), "LoweringClearStepContainerGraph");
  CheckGraphGenerally(exe_graph.get());
  EXPECT_EQ(GetNodeTypes(exe_graph->GetDirectNode()), std::set<std::string>({"Init", "Main", "DeInit"}));
  CheckAllocFreeStageMatch(exe_graph.get());
  // todo check more
}

TEST_F(GraphConverterUT, ReportErrorWhenConstDataOnMainGraph) {
  // 当前貌似没办法校验异常场景，因为CEM没有关闭选项
  auto graph = ShareGraph::BuildHostCpuDataFlowGraph();
  graph->FindNode("add1")->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCore);
  graph->TopologicalSorting();
auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).FakeWithHandleAiCore("Add", false).Build();
  ASSERT_NE(graph, nullptr);

  bg::ValueHolder::PushGraphFrame();

  auto const_data_holder = bg::FrameSelector::OnMainRoot([&]() -> std::vector<bg::ValueHolderPtr> {
    return {bg::ValueHolder::CreateConstData(static_cast<int64_t>(ConstDataType::kTypeEnd))};
  });
  EXPECT_NE(const_data_holder[0], nullptr);

  auto root_frame = bg::ValueHolder::PopGraphFrame();
  auto model_desc_holder = ModelDescHolderFaker().Build();
  auto exe_graph =
      GraphConverter().SetModelDescHolder(&model_desc_holder).ConvertComputeGraphToExecuteGraph(graph, global_data);
  ASSERT_NE(exe_graph, nullptr);

  CheckGraphGenerally(exe_graph.get());
  EXPECT_EQ(GetNodeTypes(exe_graph->GetDirectNode()), std::set<std::string>({"Init", "Main", "DeInit"}));
}

TEST_F(GraphConverterUT, ConvertComputeGraphToExecuteGraph_SaveAtomicInputsInComputeNodeInfo_StaticAicoreWithAtomic) {
  bool is_with_atomic = true;
  auto graph = ShareGraph::AicoreStaticGraph(is_with_atomic);
  graph->TopologicalSorting();

auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).FakeWithoutHandleAiCore("Add", true).Build();
  auto model_desc_holder = ModelDescHolderFaker().Build();
  auto exe_graph =
      GraphConverter().SetModelDescHolder(&model_desc_holder).ConvertComputeGraphToExecuteGraph(graph, global_data);
  ASSERT_NE(exe_graph, nullptr);
  ASSERT_EQ(3, exe_graph->GetDirectNodesSize());

  auto main_graph = exe_graph->GetAllSubgraphs()[1];
  ASSERT_NE(main_graph, nullptr);
  for (auto &node : main_graph->GetDirectNode()) {
    if (node->GetType() == kAtomicLaunchKernelType) {
      CheckAtomicInfoInComputeNodeInfo(exe_graph.get(), node, "add1_AtomicClean", 1);
      break;
    }
  }
}

TEST_F(GraphConverterUT, ConvertComputeGraphToExecuteGraph_SaveAtomicInputsInComputeNodeInfo_DynamicAicoreWithAtomic) {
  auto graph = ShareGraph::AicoreGraph();
  auto add1 = graph->FindNode("add1");
  add1->GetOpDesc()->SetOpKernelLibName(kEngineNameAiCore);
  AddCompileResult(add1, true);
  graph->TopologicalSorting();

auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).FakeWithoutHandleAiCore("Add", true).Build();
  auto model_desc_holder = ModelDescHolderFaker().Build();
  auto exe_graph =
      GraphConverter().SetModelDescHolder(&model_desc_holder).ConvertComputeGraphToExecuteGraph(graph, global_data);
  ASSERT_NE(exe_graph, nullptr);
  ASSERT_EQ(3, exe_graph->GetDirectNodesSize());

  auto main_graph = exe_graph->GetAllSubgraphs()[1];
  ASSERT_NE(main_graph, nullptr);
  for (auto &node : main_graph->GetDirectNode()) {
    if (node->GetType() == kAtomicLaunchKernelType) {
      CheckAtomicInfoInComputeNodeInfo(exe_graph.get(), node, "add1_AtomicClean", 1);
      break;
    }
  }
}

TEST_F(GraphConverterUT, ConvertWithPartitionedCall_CheckInnerDataLowerResult_Success) {
  auto graph = ShareGraph::BuildWithInnerDataSubgraph();
  graph->TopologicalSorting();
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto faker = GlobalDataFaker(root_model);
  auto global_data = faker.Build();

  auto model_desc_holder = ModelDescHolderFaker().Build();
  auto exe_graph =
      GraphConverter().SetModelDescHolder(&model_desc_holder).ConvertComputeGraphToExecuteGraph(graph, global_data);
  ASSERT_NE(exe_graph, nullptr);
  auto data_a = graph->FindNode("data_a");
  const auto *const_data_a_result = data_a->GetOpDesc()->GetExtAttr<gert::PlacedLoweringResult>(kLoweringResult);
  auto *data_a_result = const_cast<PlacedLoweringResult *>(const_data_a_result);
  auto data_a_out_result = data_a_result->GetOutputResult(global_data, 0, {kOnDeviceHbm, bg::kMainStream}, false);
  auto parent_node = graph->FindNode("PartitionedCall");
  ASSERT_NE(parent_node, nullptr);
  auto root_compute_graph = ge::GraphUtils::FindRootGraph(parent_node->GetOwnerComputeGraph());
  ASSERT_NE(root_compute_graph, nullptr);
  auto sub_graph = root_compute_graph->GetSubgraph(parent_node->GetOpDesc()->GetSubgraphInstanceName(0U));

  auto data_1 = sub_graph->FindNode("data_1");
  ASSERT_NE(data_1, nullptr);
  const auto *const_data_1_result = data_1->GetOpDesc()->GetExtAttr<gert::PlacedLoweringResult>(kLoweringResult);
  auto *data_1_result = const_cast<PlacedLoweringResult *>(const_data_1_result);
  ASSERT_EQ(data_1_result->GetResult()->out_addrs.size(), 1);
  EXPECT_EQ(data_a_out_result->address, data_1_result->GetResult()->out_addrs[0]);
}

TEST_F(GraphConverterUT, MultiStream_LoweringTwoNodeCrossStream) {
  int64_t stream_num = 1;
  int64_t event_num = 0;
  auto graph = ShareGraph::MultiStreamTwoNodeGraph(stream_num, event_num);
  graph->TopologicalSorting();
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data =
      GlobalDataFaker(root_model).FakeWithHandleAiCore("Add", false).FakeWithHandleAiCore("Relu", false).Build();
  ASSERT_NE(graph, nullptr);
  auto model_desc_holder = ModelDescHolderFaker().Build(stream_num, event_num);
  auto exe_graph =
      GraphConverter().SetModelDescHolder(&model_desc_holder).ConvertComputeGraphToExecuteGraph(graph, global_data);
  ASSERT_NE(exe_graph, nullptr);
  ASSERT_EQ(exe_graph->TopologicalSorting(), ge::GRAPH_SUCCESS);
  ge::DumpGraph(exe_graph.get(), "MultiStream_LoweringTwoNodeCrossStream");

  CheckGraphGenerally(exe_graph.get());
  EXPECT_EQ(GetNodeTypes(exe_graph->GetDirectNode()), std::set<std::string>({"Init", "Main", "DeInit"}));
  CheckAllocFreeStageMatch(exe_graph.get());

  auto init_graph =
      ge::FastNodeUtils::GetSubgraphFromNode(ge::ExecuteGraphUtils::FindFirstNodeMatchType(exe_graph.get(), "Init"), 0);
  EXPECT_EQ(ExeGraphSummaryChecker(init_graph)
                .DirectNodeTypesOnlyCareAbout({{"CreateL2Allocators", 1}, {"CreateGertEvents", 1}}),
            "success");

  auto main_graph =
      ge::FastNodeUtils::GetSubgraphFromNode(ge::ExecuteGraphUtils::FindFirstNodeMatchType(exe_graph.get(), "Main"), 0);
  EXPECT_EQ(ExeGraphSummaryChecker(main_graph)
                .DirectNodeTypesOnlyCareAbout(
                    {{"Data", (2 + 4)},
                     {"AccessMemCrossStream", 2},
                     {"SendEvents", 2},
                     {"WaitEvents", 2},
                     {"SelectL2Allocator", 2},  // stream0_device_allocator, stream1_device_allocator
                     {"SplitRtStreams", 1}}),
            "success");
  // check AccessMemCrossStream_relu connection
  auto access_cross_stream_relu = ge::ExecuteGraphUtils::FindFirstNodeMatchType(main_graph, "AccessMemCrossStream");
  ASSERT_EQ(FastNodeTopoChecker(access_cross_stream_relu).StrictConnectFrom({{"AllocMemHbm", 0}, {"Const", 0}}, true),
            "success");
  ASSERT_EQ(FastNodeTopoChecker(access_cross_stream_relu)
                .StrictConnectTo(0, {{"LaunchKernelWithHandle", 13}, {"FreeMemory", 0}}),
            "success");
  ASSERT_EQ(FastNodeTopoChecker(access_cross_stream_relu).StrictConnectTo(-1, {{"FreeMemory", -1}}), "success");
  // check SendEvent connection
  auto send_add = ge::ExecuteGraphUtils::FindFirstNodeMatchType(main_graph, "SendEvents");
  ASSERT_EQ(FastNodeTopoChecker(send_add).StrictConnectFrom({{"SplitRtStreams", 0},    // dst_stream
                                                             {"Const", 0},             // event ids
                                                             {"CreateGertEvents", 0},  // all gert events
                                                             {"Data", 0},              // rt_events
                                                             {"SelectL2Allocator", 0},
                                                             {"LaunchKernelWithHandle", -1}},
                                                            true),
            "success");
  ASSERT_EQ(FastNodeTopoChecker(send_add).StrictConnectTo(
                -1, {{"LaunchKernelWithHandle", -1}, {"WaitEvents", -1}, {"SendEvents", -1}}),
            "success");

  // check WaitEvents connection
  auto wait_relu = ge::ExecuteGraphUtils::FindFirstNodeMatchType(main_graph, "WaitEvents");
  ASSERT_EQ(FastNodeTopoChecker(wait_relu).StrictConnectFrom({{"SplitRtStreams", 1},
                                                              {"Const", 0},
                                                              {"CreateGertEvents", 0},
                                                              {"Data", 0},
                                                              {"SelectL2Allocator", 0},
                                                              {"LaunchKernelWithHandle", -1},
                                                              {"SendEvents", -1}},
                                                             true),
            "success");
  ASSERT_EQ(FastNodeTopoChecker(wait_relu).StrictConnectTo(
                -1, {{"LaunchKernelWithHandle", -1}, {"EnsureTensorAtOutMemory", -1}, {"WaitEvents", -1}}),
            "success");
}
/*
 * check H2D dst stream is same with dst node
 */
TEST_F(GraphConverterUT, MultiStream_LoweringTwoNodeCrossStreamWithValueDepend) {
  int64_t stream_num, event_num;
  auto graph = ShareGraph::MultiStreamWithHostMemAccessCrossStream(stream_num, event_num);
  graph->TopologicalSorting();
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data =
      GlobalDataFaker(root_model).FakeWithHandleAiCore("Add", false).Build();
  ASSERT_NE(graph, nullptr);
  auto model_desc_holder = ModelDescHolderFaker().Build(stream_num, event_num);
  auto exe_graph =
      GraphConverter().SetModelDescHolder(&model_desc_holder).ConvertComputeGraphToExecuteGraph(graph, global_data);
  ASSERT_NE(exe_graph, nullptr);
  ASSERT_EQ(exe_graph->TopologicalSorting(), ge::GRAPH_SUCCESS);
  ge::DumpGraph(exe_graph.get(), "MultiStream_LoweringTwoNodeCrossStreamWithValueDepend");

  CheckGraphGenerally(exe_graph.get());
  EXPECT_EQ(GetNodeTypes(exe_graph->GetDirectNode()), std::set<std::string>({"Init", "Main", "DeInit"}));
  CheckAllocFreeStageMatch(exe_graph.get());

  auto init_graph =
      ge::FastNodeUtils::GetSubgraphFromNode(ge::ExecuteGraphUtils::FindFirstNodeMatchType(exe_graph.get(), "Init"), 0);
  EXPECT_EQ(ExeGraphSummaryChecker(init_graph)
                .DirectNodeTypesOnlyCareAbout({{"CreateL2Allocators", 1}, {"CreateGertEvents", 1}}),
            "success");

  auto main_graph =
      ge::FastNodeUtils::GetSubgraphFromNode(ge::ExecuteGraphUtils::FindFirstNodeMatchType(exe_graph.get(), "Main"), 0);
  EXPECT_EQ(ExeGraphSummaryChecker(main_graph)
                .DirectNodeTypesOnlyCareAbout({{"Data", (2 + 4)},
                                               {"AccessMemCrossStream", 2},
                                               {"SendEvents", 3},
                                               {"WaitEvents", 2},
                                               {"SelectL2Allocator", 2},
                                               {"SplitRtStreams", 1}}),
            "success");
  // check H2D stream_id connection
  auto h2d = ge::ExecuteGraphUtils::FindFirstNodeMatchType(main_graph,
                                                           "CopyFlowLaunch");  // CopyH2D optimized to CopyFlowLaunch
  auto stream_input_node = h2d->GetInDataEdgeByIndex(static_cast<int32_t>(kernel::CopyFlowLaunchInputs::kStream))->src;
  auto logic_stream_inner_data = stream_input_node->GetInDataEdgeByIndex(0)->src;
  ASSERT_EQ(logic_stream_inner_data->GetType(), "Data");
  // todo
  /*int32_t index;
  ASSERT_TRUE(ge::AttrUtils::GetInt(logic_stream_inner_data->GetOpDesc(), "index", index));
  auto main_node = logic_stream_inner_data->GetOwnerComputeGraph()->GetParentNode();
  auto out_index_in_init = main_node->GetInDataAnchor(index)->GetPeerOutAnchor()->GetIdx();
  auto init_netoutput = init_graph->FindFirstNodeMatchType("InnerNetOutput");
  auto logic_stream_const = init_netoutput->GetInDataAnchor(out_index_in_init)->GetPeerOutAnchor()->GetOwnerNode();
  ASSERT_NE(logic_stream_const, nullptr);

  ge::Buffer buffer;
  ASSERT_TRUE(ge::AttrUtils::GetZeroCopyBytes(logic_stream_const->GetOpDesc(), "value", buffer));
  ASSERT_EQ(buffer.GetSize(), sizeof(int64_t));
  EXPECT_EQ(*reinterpret_cast<int64_t *>(buffer.GetData()), 1); // add node logic_stream_id is 1*/
}

TEST_F(GraphConverterUT, MultiStream_LoweringFristEventSync) {
  int64_t stream_num, event_num;
  auto graph = ShareGraph::MultiStreamGraphWithFirstEventSyncGraph(stream_num, event_num);
  graph->TopologicalSorting();
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data =
      GlobalDataFaker(root_model).FakeWithHandleAiCore("Add", false).Build();
  ASSERT_NE(graph, nullptr);
  auto model_desc_holder = ModelDescHolderFaker().Build(stream_num, event_num);
  auto exe_graph =
      GraphConverter().SetModelDescHolder(&model_desc_holder).ConvertComputeGraphToExecuteGraph(graph, global_data);
  ASSERT_NE(exe_graph, nullptr);
  ge::DumpGraph(exe_graph.get(), "MultiStream_LoweringFristEventSync");
  ASSERT_EQ(exe_graph->TopologicalSorting(), ge::GRAPH_SUCCESS);

  CheckGraphGenerally(exe_graph.get());
  EXPECT_EQ(GetNodeTypes(exe_graph->GetDirectNode()), std::set<std::string>({"Init", "Main", "DeInit"}));
  CheckAllocFreeStageMatch(exe_graph.get());

  auto init_graph =
      ge::FastNodeUtils::GetSubgraphFromNode(ge::ExecuteGraphUtils::FindFirstNodeMatchType(exe_graph.get(), "Init"), 0);
  EXPECT_EQ(ExeGraphSummaryChecker(init_graph)
                .DirectNodeTypesOnlyCareAbout({{"CreateL2Allocators", 1}, {"CreateGertEvents", 1}}),
            "success");

  auto main_graph =
      ge::FastNodeUtils::GetSubgraphFromNode(ge::ExecuteGraphUtils::FindFirstNodeMatchType(exe_graph.get(), "Main"), 0);
  EXPECT_EQ(ExeGraphSummaryChecker(main_graph)
                .DirectNodeTypesOnlyCareAbout({{"Data", (2 + 4)},
                                               {"AccessMemCrossStream", 1},
                                               {"SendEvents", 2},
                                               {"WaitEvents", 2},
                                               {"SelectL2Allocator", 2},
                                               {"SplitRtStreams", 1}}),
            "success");
}

TEST_F(GraphConverterUT, MultiStream_LoweringLastEventSync) {
  int64_t stream_num, event_num;
  auto graph = ShareGraph::MultiStreamGraphWithLastEventSyncGraph(stream_num, event_num);
  graph->TopologicalSorting();
  AiCpuTfTaskDefFaker aicpu_task_def_faker;
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model)
                         .FakeWithHandleAiCore("Relu", false)
                         .AddTaskDef("Add", aicpu_task_def_faker.SetNeedMemcpy(true))
                         .Build();
  ASSERT_NE(graph, nullptr);
  auto model_desc_holder = ModelDescHolderFaker().Build(stream_num, event_num);
  auto exe_graph =
      GraphConverter().SetModelDescHolder(&model_desc_holder).ConvertComputeGraphToExecuteGraph(graph, global_data);
  ASSERT_NE(exe_graph, nullptr);
  ge::DumpGraph(exe_graph.get(), "MultiStream_LoweringLastEventSync");
  ASSERT_EQ(exe_graph->TopologicalSorting(), ge::GRAPH_SUCCESS);

  CheckGraphGenerally(exe_graph.get());
  EXPECT_EQ(GetNodeTypes(exe_graph->GetDirectNode()), std::set<std::string>({"Init", "Main", "DeInit"}));
  CheckAllocFreeStageMatch(exe_graph.get());

  auto init_graph =
      ge::FastNodeUtils::GetSubgraphFromNode(ge::ExecuteGraphUtils::FindFirstNodeMatchType(exe_graph.get(), "Init"), 0);
  EXPECT_EQ(ExeGraphSummaryChecker(init_graph)
                .DirectNodeTypesOnlyCareAbout({{"CreateL2Allocators", 1}, {"CreateGertEvents", 1}}),
            "success");

  auto main_graph =
      ge::FastNodeUtils::GetSubgraphFromNode(ge::ExecuteGraphUtils::FindFirstNodeMatchType(exe_graph.get(), "Main"), 0);
  EXPECT_EQ(ExeGraphSummaryChecker(main_graph)
                .DirectNodeTypesOnlyCareAbout({{"Data", (2 + 4)},
                                               {"AccessMemCrossStream", 2},
                                               {"SendEvents", 3},
                                               {"WaitEvents", 2},
                                               {"SelectL2Allocator", 2},
                                               {"SplitRtStreams", 1}}),
            "success");
  // check connection
}
}  // namespace gert
