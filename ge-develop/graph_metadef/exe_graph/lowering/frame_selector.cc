/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "exe_graph/lowering/frame_selector.h"

#include "common/checker.h"
#include "exe_graph/lowering/graph_frame.h"
#include "exe_graph/lowering/value_holder.h"
#include "exe_graph/runtime/execute_graph_types.h"
#include "graph_metadef/graph/debug/ge_util.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/execute_graph_utils.h"
#include "scoped_current_frame.h"
#include "value_holder_inner.h"

namespace gert {
namespace bg {
namespace {
const int64_t kControlAnchorIdx = -1;
ge::EdgeSrcEndpoint CreateInnerData(ge::ExecuteGraph *graph, const GraphFrame &graph_frame,
                                    const size_t index) {
  auto op_desc = ge::ComGraphMakeShared<ge::OpDesc>(ValueHolder::GenerateNodeName(kInnerData, graph_frame), kInnerData);
  GE_ASSERT_NOTNULL(op_desc);
  GE_ASSERT_SUCCESS(op_desc->AddOutputDesc(ge::GeTensorDesc()));
  GE_ASSERT_TRUE(ge::AttrUtils::SetInt(op_desc, "index", index));
  auto node = graph->AddNode(op_desc);
  GE_ASSERT_NOTNULL(node);
  return {node, 0};
}

ge::FastNode *GetOrCreateInnerNetOutput(const GraphFrame &frame) {
  auto netoutput = ge::ExecuteGraphUtils::FindFirstNodeMatchType(frame.GetExecuteGraph().get(), kInnerNetOutput);
  if (netoutput != nullptr) {
    return netoutput;
  }
  return ValueHolder::AddNode(kInnerNetOutput, 0U, 0U, frame);
}

ge::graphStatus MoveGuardersToDeInit(ge::FastNode *init_node, const GraphFrame &root_frame,
                                     const std::vector<std::pair<ValueHolderPtr, size_t>> &guarders_and_out_index) {
  const auto de_init_node =
      ge::ExecuteGraphUtils::FindFirstNodeMatchType(root_frame.GetExecuteGraph().get(),
                                                    GetExecuteGraphTypeStr(ExecuteGraphType::kDeInit));
  GE_ASSERT_NOTNULL(de_init_node);
  auto de_init_graph = ge::FastNodeUtils::GetSubgraphFromNode(de_init_node, 0U);
  GE_ASSERT_NOTNULL(de_init_graph);

  auto index = de_init_node->GetDataInNum();
  GE_ASSERT_SUCCESS(ge::FastNodeUtils::AppendInputEdgeInfo(de_init_node, index + guarders_and_out_index.size()));

  auto init_graph = init_node->GetExtendInfo()->GetOwnerGraphBarePtr();
  GE_ASSERT_NOTNULL(init_graph);
  for (size_t i = 0U; i < guarders_and_out_index.size(); ++i) {
    auto guarder_node = guarders_and_out_index[i].first->GetFastNode();
    GE_ASSERT_NOTNULL(guarder_node);
    GE_ASSERT_SUCCESS(
        ge::ExecuteGraphUtils::MoveNodeToGraph(guarder_node, de_init_graph));
    GE_ASSERT_NOTNULL(init_graph->AddEdge(init_node,
                                          static_cast<int32_t>(guarders_and_out_index[i].second), de_init_node,
                                          static_cast<int32_t>(index + i)));
    auto src_end_point = CreateInnerData(de_init_graph, root_frame, index + i);
    GE_ASSERT_NOTNULL(src_end_point.node);
    GE_ASSERT_NOTNULL(de_init_graph->AddEdge(src_end_point.node, src_end_point.index,
                                             guarders_and_out_index[i].first->GetFastNode(), 0));
  }

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus GetNewOutIndexes(const std::vector<ValueHolderPtr> &init_graph_outputs, const size_t start_index,
                                 size_t &new_out_num, std::map<ValueHolderPtr, int64_t> &graph_outputs_to_out_index) {
  new_out_num = 0U;
  for (const auto &output : init_graph_outputs) {
    GE_ASSERT_NOTNULL(output, "Failed to construct on init graph, the graph builder return nullptr");
    const auto node = output->GetFastNode();
    GE_ASSERT_NOTNULL(node);
    const auto index = output->GetOutIndex();
    if (index < 0) {
      graph_outputs_to_out_index[output] = kControlAnchorIdx;
      continue;
    }
    int32_t out_index = -1;
    for (const auto edge : node->GetOutEdgesRefByIndex(index)) {
      if (edge == nullptr) {
        continue;
      }
      GE_ASSERT_NOTNULL(edge->dst);
      if (IsTypeInnerNetOutput(edge->dst->GetTypePtr())) {
        out_index = edge->dst_input;
        break;
      }
    }
    if (out_index < 0) {
      graph_outputs_to_out_index[output] = start_index + (new_out_num++);
    } else {
      graph_outputs_to_out_index[output] = static_cast<size_t>(out_index);
    }
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus ConnectSubGraphOut(ge::FastNode *parent_node, GraphFrame &sub_frame,
                                   const std::vector<ValueHolderPtr> &sub_graph_outputs,
                                   std::vector<std::pair<ValueHolderPtr, size_t>> &guarders_and_out_index,
                                   std::vector<ValueHolderPtr> &parent_node_outputs) {
  auto netoutput = GetOrCreateInnerNetOutput(sub_frame);
  GE_ASSERT_NOTNULL(netoutput);
  const auto index = netoutput->GetDataInNum();

  size_t new_out_num;
  std::map<ValueHolderPtr, int64_t> graph_outputs_to_out_index;
  GE_ASSERT_SUCCESS(GetNewOutIndexes(sub_graph_outputs, index, new_out_num, graph_outputs_to_out_index));

  GE_ASSERT_SUCCESS(ge::FastNodeUtils::AppendInputEdgeInfo(netoutput, index + new_out_num));
  guarders_and_out_index.reserve(sub_graph_outputs.size());
  parent_node_outputs.reserve(sub_graph_outputs.size());
  for (const auto &holder : sub_graph_outputs) {
    GE_ASSERT_NOTNULL(holder);
    const auto &out_index = graph_outputs_to_out_index.at(holder);
    auto node = holder->GetFastNode();
    GE_ASSERT_NOTNULL(node);
    auto graph = node->GetExtendInfo()->GetOwnerGraphBarePtr();
    GE_ASSERT_NOTNULL(graph);
    if (out_index >= static_cast<int64_t>(index)) {
      GE_ASSERT_NOTNULL(graph->AddEdge(node, holder->GetOutIndex(), netoutput, out_index));
    } else if (out_index == kControlAnchorIdx) {
      GE_ASSERT_NOTNULL(graph->AddEdge(node, ge::kControlEdgeIndex,
                                       netoutput, ge::kControlEdgeIndex));
    }

    auto guarder = holder->GetGuarder();
    if (guarder != nullptr) {
      guarders_and_out_index.emplace_back(guarder, out_index);
    }

    parent_node_outputs.emplace_back(holder->CreateMateFromNode(parent_node, static_cast<int32_t>(out_index),
                                                                ValueHolder::ValueHolderType::kOutput));
  }

  GE_ASSERT_SUCCESS(ge::FastNodeUtils::AppendOutputEdgeInfo(parent_node, index + new_out_num));

  for (size_t i = 0U; i < sub_graph_outputs.size(); ++i) {
    parent_node_outputs[i]->SetPlacement(sub_graph_outputs[i]->GetPlacement());
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus ConnectOut(ge::FastNode *init_node, GraphFrame &root_frame, GraphFrame &init_frame,
                           const std::vector<ValueHolderPtr> &init_graph_outputs,
                           std::vector<ValueHolderPtr> &init_node_outputs) {
  std::vector<std::pair<ValueHolderPtr, size_t>> guarders_and_out_index;
  GE_ASSERT_SUCCESS(
      ConnectSubGraphOut(init_node, init_frame, init_graph_outputs, guarders_and_out_index, init_node_outputs));
  if (!guarders_and_out_index.empty()) {
    GE_ASSERT_SUCCESS(MoveGuardersToDeInit(init_node, root_frame, guarders_and_out_index));
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus LoweringToSubgraph(const std::function<std::vector<ValueHolderPtr>()> &builder,
                                   ValueHolderPtr partition_call_holder, GraphFrame *sub_graph_frame,
                                   std::vector<ValueHolderPtr> &parent_node_outputs) {
  const ScopedCurrentFrame frame_guarder(sub_graph_frame);
  auto outputs = builder();
  partition_call_holder->AppendOutputs<ValueHolder>(outputs.size());

  std::vector<std::pair<ValueHolderPtr, size_t>> guarders_and_out_index;
  auto partitioned_call = const_cast<ge::FastNode *>(sub_graph_frame->GetExecuteGraph()->GetParentNodeBarePtr());
  GE_ASSERT_GRAPH_SUCCESS(
      ConnectSubGraphOut(partitioned_call, *sub_graph_frame, outputs, guarders_and_out_index, parent_node_outputs));
  return ge::GRAPH_SUCCESS;
}

bool InitStageIdsToPartitionedCalls(const char *attr_name, std::vector<ValueHolderPtr> &stage_ids_to_pcall) {
  if (attr_name == kStageIdsToFirstPartitionedCall) {
    stage_ids_to_pcall.resize(static_cast<size_t>(OnMainRootFirstExecStage::kStageSize));
  } else if (attr_name == kStageIdsToLastPartitionedCall) {
    stage_ids_to_pcall.resize(static_cast<size_t>(OnMainRootLastExecStage::kStageSize));
  } else {
    return false;
  }
  return true;
}

std::vector<ValueHolderPtr> GetOrCreateAllPartitionedCalls(const ge::ExecuteGraph *exe_graph,
                                                           const char *stage_attr_name) {
  std::vector<ValueHolderPtr> tmp_stage_ids_to_pcall;
  auto stage_ids_to_partitioned_calls = exe_graph->GetExtAttr<std::vector<ValueHolderPtr>>(stage_attr_name);
  if (stage_ids_to_partitioned_calls == nullptr) {
    GE_ASSERT_TRUE(InitStageIdsToPartitionedCalls(stage_attr_name, tmp_stage_ids_to_pcall));
    return tmp_stage_ids_to_pcall;
  } else {
    return *stage_ids_to_partitioned_calls;
  }
}

ValueHolderPtr GetOrCreatePartitionedCallHolder(std::vector<ValueHolderPtr> &stage_ids_to_pcalls, size_t stage_id) {
  GE_ASSERT_TRUE(stage_ids_to_pcalls.size() > stage_id);
  auto partition_call_holder = stage_ids_to_pcalls[stage_id];
  if (partition_call_holder != nullptr) {
    return partition_call_holder;
  }
  return ValueHolder::CreateVoid<ValueHolder>("PartitionedCall", {});
}

GraphFrame *PushPartitionedCallSubFrame(bg::ValueHolderPtr &partition_call_holder) {
  GE_ASSERT_NOTNULL(partition_call_holder->GetFastNode());
  auto sub_graph = ge::FastNodeUtils::GetSubgraphFromNode(partition_call_holder->GetFastNode(), 0U);
  if (sub_graph == nullptr) {
    return ValueHolder::PushGraphFrame(partition_call_holder, "exec_sub_graph");
  }

  GraphFrame *cur_frame = ValueHolder::GetCurrentFrame();
  GE_ASSERT_NOTNULL(cur_frame);
  std::unique_ptr<GraphFrame> sub_graph_frame_holder =
      ge::ComGraphMakeUnique<GraphFrame>(sub_graph->shared_from_this(), *cur_frame);
  GE_ASSERT_NOTNULL(sub_graph_frame_holder);
  sub_graph_frame_holder->SetCurrentComputeNode(cur_frame->GetCurrentComputeNode());
  return ValueHolder::PushGraphFrame(sub_graph_frame_holder.release());
}

std::vector<ValueHolderPtr> OnMainRootPartitionedCall(
    const std::function<std::vector<bg::ValueHolderPtr>()> &partition_call_builder, const char *attr_name,
    size_t stage_id) {
  GE_ASSERT_NOTNULL(partition_call_builder);
  GE_ASSERT_TRUE(GetGraphFrames().size() > 1U);
  GraphFrame *current_frame = (GetGraphFrames().begin() + 1)->get();
  GE_ASSERT_EQ(current_frame->GetExecuteGraph()->GetParentNodeBarePtr()->GetType(), "Main");
  const ScopedCurrentFrame main_frame_guarder(current_frame);

  std::vector<ValueHolderPtr> stage_ids_to_pcall =
      GetOrCreateAllPartitionedCalls(current_frame->GetExecuteGraph().get(), attr_name);
  GE_ASSERT_TRUE(stage_ids_to_pcall.size() > stage_id, "Stage_ids_2_partitioncall size %zu, stage_id is %zu",
                 stage_ids_to_pcall.size(), stage_id);
  ValueHolderPtr partition_call_holder = GetOrCreatePartitionedCallHolder(stage_ids_to_pcall, stage_id);
  GE_ASSERT_NOTNULL(partition_call_holder);
  GraphFrame *sub_graph_frame = PushPartitionedCallSubFrame(partition_call_holder);
  GE_ASSERT_NOTNULL(sub_graph_frame);

  std::vector<ValueHolderPtr> parent_node_outputs;
  GE_ASSERT_GRAPH_SUCCESS(
      LoweringToSubgraph(partition_call_builder, partition_call_holder, sub_graph_frame, parent_node_outputs));

  stage_ids_to_pcall[stage_id] = partition_call_holder;
  current_frame->GetExecuteGraph()->SetExtAttr<std::vector<ValueHolderPtr>>(attr_name, stage_ids_to_pcall);
  ValueHolder::PopGraphFrame();
  return parent_node_outputs;
}

ge::graphStatus OnDeInitGraph(const std::function<std::vector<ValueHolderPtr>()> &builder,
                              std::vector<ValueHolderPtr> &de_init_nodss) {
  GE_ASSERT_NOTNULL(builder, "Failed to do frame selection, the builder is nullptr");
  GE_ASSERT_TRUE(!GetGraphFrames().empty(), "Failed to do frame selection, there is no root-frame exists");

  const auto root_frame = GetGraphFrames().begin()->get();
  GE_ASSERT_NOTNULL(root_frame, "Failed to find the root frame");

  // check if the main_frame is correct
  const auto root_graph = root_frame->GetExecuteGraph();
  GE_ASSERT_NOTNULL(root_graph, "Failed to find the root graph");
  const auto de_init_node =
    ge::ExecuteGraphUtils::FindFirstNodeMatchType(root_graph.get(), GetExecuteGraphTypeStr(ExecuteGraphType::kDeInit));
  GE_ASSERT_NOTNULL(de_init_node, "Failed to find the de_init node");
  const auto de_init_graph = ge::FastNodeUtils::GetSubgraphFromNode(de_init_node, 0U);
  GE_ASSERT_NOTNULL(de_init_graph, "Failed to find the DeInit graph from de_init node %s",
                    de_init_node->GetName().c_str());

  auto tmp_de_init_frame = ge::ComGraphMakeUnique<GraphFrame>(de_init_graph->shared_from_this(), *root_frame);
  GE_ASSERT_NOTNULL(tmp_de_init_frame);

  tmp_de_init_frame->SetCurrentComputeNode(ValueHolder::GetCurrentFrame()->GetCurrentComputeNode());

  const ScopedCurrentFrame frame_guarder(tmp_de_init_frame.get());

  de_init_nodss = builder();

  return ge::GRAPH_SUCCESS;
}
}  // namespace
std::vector<ValueHolderPtr> FrameSelector::OnMainRoot(const std::function<std::vector<ValueHolderPtr>()> &builder) {
  if (builder == nullptr || GetGraphFrames().empty()) {
    return {};
  }
  std::vector<ValueHolderPtr> outputs;
  if (OnMainRoot(builder, outputs) != ge::GRAPH_SUCCESS) {
    GELOGW("Compatible mode, the air code is not the newest.");
    const ScopedCurrentFrame frame_guarder(GetGraphFrames().front().get());
    outputs = builder();
  }
  return outputs;
}

ge::graphStatus FrameSelector::OnMainRoot(const std::function<std::vector<ValueHolderPtr>()> &builder,
                                          std::vector<ValueHolderPtr> &outputs) {
  GE_ASSERT_NOTNULL(builder, "Failed to do frame selection, the builder is nullptr");
  // 栈底是root-frame，向上是main-frame，因此栈中至少有两个元素
  GE_ASSERT_TRUE(GetGraphFrames().size() > 1U, "Failed to do frame selection, there is no main-frame exists");

  const auto root_frame = GetGraphFrames().begin()->get();
  GE_ASSERT_NOTNULL(root_frame, "Failed to find the root frame");
  auto main_frame = (GetGraphFrames().begin() + 1)->get();
  GE_ASSERT_NOTNULL(main_frame, "Failed to find the main frame");

  // check if the main_frame is correct
  const auto root_graph = root_frame->GetExecuteGraph();
  GE_ASSERT_NOTNULL(root_graph, "Failed to find the root graph");
  const auto main_node =
    ge::ExecuteGraphUtils::FindFirstNodeMatchType(root_graph.get(), GetExecuteGraphTypeStr(ExecuteGraphType::kMain));
  GE_ASSERT_NOTNULL(main_node, "Failed to find the main node");
  const auto main_graph = ge::FastNodeUtils::GetSubgraphFromNode(main_node, 0U);
  GE_ASSERT_TRUE(main_graph == main_frame->GetExecuteGraph().get(), "Failed to find the main frame");

  const ScopedCurrentFrame frame_guarder(main_frame);
  outputs = builder();
  return ge::GRAPH_SUCCESS;
}

std::vector<ValueHolderPtr> FrameSelector::OnMainRootFirst(
    const std::function<std::vector<bg::ValueHolderPtr>()> &builder) {
  return OnMainRootPartitionedCall(builder, kStageIdsToFirstPartitionedCall,
                                   static_cast<size_t>(OnMainRootFirstExecStage::kFirstEventSyncStage));
}

std::vector<ValueHolderPtr> FrameSelector::OnDeInitRoot(const std::function<std::vector<ValueHolderPtr>()> &builder) {
  if (builder == nullptr || GetGraphFrames().empty()) {
    return {};
  }
  std::vector<ValueHolderPtr> de_init_nodes;
  if (OnDeInitGraph(builder, de_init_nodes) != ge::GRAPH_SUCCESS) {
    return {};
  }
  return de_init_nodes;
}

std::vector<ValueHolderPtr> FrameSelector::OnInitRoot(const std::function<std::vector<ValueHolderPtr>()> &builder) {
  std::vector<ValueHolderPtr> init_graph_outputs;
  std::vector<ValueHolderPtr> init_node_outputs;
  const auto ret = OnInitRoot(builder, init_graph_outputs, init_node_outputs);
  if (ret != ge::GRAPH_SUCCESS) {
    return {};
  }
  return init_node_outputs;
}

ge::graphStatus FrameSelector::OnInitRoot(const std::function<std::vector<ValueHolderPtr>()> &builder,
                                          std::vector<ValueHolderPtr> &init_graph_outputs,
                                          std::vector<ValueHolderPtr> &init_node_outputs) {
  GE_ASSERT_NOTNULL(builder, "Failed to do frame selection, the builder is nullptr");
  GE_ASSERT_TRUE(!GetGraphFrames().empty(), "Failed to do frame selection, there is no root-frame exists");

  const auto root_frame = GetGraphFrames().begin()->get();
  GE_ASSERT_NOTNULL(root_frame, "Failed to find the root frame");

  const auto init_node =
      ge::ExecuteGraphUtils::FindFirstNodeMatchType(root_frame->GetExecuteGraph().get(),
                                                    GetExecuteGraphTypeStr(ExecuteGraphType::kInit));
  GE_ASSERT_NOTNULL(init_node, "Failed to find the Init node from root graph");
  auto init_graph = ge::FastNodeUtils::GetSubgraphFromNode(init_node, 0U);
  GE_ASSERT_NOTNULL(init_graph, "Failed to find the Init graph from init node %s", init_node->GetNamePtr());

  auto tmp_init_frame = ge::ComGraphMakeUnique<GraphFrame>(init_graph->shared_from_this(), *root_frame);
  GE_ASSERT_NOTNULL(tmp_init_frame);

  tmp_init_frame->SetCurrentComputeNode(ValueHolder::GetCurrentFrame()->GetCurrentComputeNode());
  const ScopedCurrentFrame frame_guarder(tmp_init_frame.get());
  init_graph_outputs = builder();
  if (!init_graph_outputs.empty()) {
    return ConnectOut(init_node, *root_frame, *tmp_init_frame, init_graph_outputs, init_node_outputs);
  }
  return ge::GRAPH_SUCCESS;
}

ValueHolderPtr FrameSelector::OnMainRootLast(const std::function<bg::ValueHolderPtr()> &builder) {
  if (builder == nullptr || GetGraphFrames().empty()) {
    return nullptr;
  }
  GraphFrame *current_frame = nullptr;
  if (GetGraphFrames().size() > 1U) {
    current_frame = (GetGraphFrames().begin() + 1)->get();
  } else {
    current_frame = GetGraphFrames().begin()->get();
  }
  GE_ASSERT_NOTNULL(current_frame);
  const ScopedCurrentFrame frame_guarder(current_frame);
  auto output = builder();
  GetCurrentFrame()->SetLastExecNode(output);
  return output;
}

std::vector<ValueHolderPtr> FrameSelector::OnMainRootLastEventSync(
    const std::function<std::vector<bg::ValueHolderPtr>()> &builder) {
  return OnMainRootPartitionedCall(builder, kStageIdsToLastPartitionedCall,
                                   static_cast<size_t>(OnMainRootLastExecStage::kLastEventSyncStage));
}

std::vector<ValueHolderPtr> FrameSelector::OnMainRootLastResourceClean(
    const std::function<std::vector<bg::ValueHolderPtr>()> &builder) {
  return OnMainRootPartitionedCall(builder, kStageIdsToLastPartitionedCall,
                                   static_cast<size_t>(OnMainRootLastExecStage::kLastResourceClean));
}

ValueHolderPtr HolderOnInit(const ValueHolderPtr &holder) {
  GE_ASSERT_NOTNULL(holder);
  const auto index = holder->GetOutIndex();
  GE_ASSERT_TRUE(index >= 0, "The holder is a ctrl holder");

  const auto holder_node = holder->GetFastNode();
  GE_ASSERT_NOTNULL(holder_node);
  if (strcmp(holder_node->GetTypePtr(), GetExecuteGraphTypeStr(ExecuteGraphType::kInit)) == 0) {
    const auto init_graph = ge::FastNodeUtils::GetSubgraphFromNode(holder_node, 0U);
    GE_ASSERT_NOTNULL(init_graph);
    const auto netoutput = ge::ExecuteGraphUtils::FindFirstNodeMatchType(init_graph, kInnerNetOutput);
    GE_ASSERT_NOTNULL(netoutput, "Can not find the InnerNetOutput node on the Init graph");
    
    auto edge = netoutput->GetInDataEdgeByIndex(index);
    GE_ASSERT_NOTNULL(edge, "The InnerNetOutput does not have the in edge %d", index);
    auto src_node = edge->src;
    GE_ASSERT_NOTNULL(src_node);
    return holder->CreateMateFromNode(src_node, edge->src_output, ValueHolder::ValueHolderType::kOutput);
  }

  const auto holder_graph = holder_node->GetExtendInfo()->GetOwnerGraphBarePtr();
  GE_ASSERT_NOTNULL(holder_graph);
  const auto parent_node = holder_graph->GetParentNodeBarePtr();
  GE_ASSERT_NOTNULL(parent_node, "The node %s is not on the Root graph", holder_node->GetNamePtr());
  if (strcmp(parent_node->GetTypePtr(), GetExecuteGraphTypeStr(ExecuteGraphType::kInit)) == 0) {
    return holder;
  }
  return nullptr;
}
}  // namespace bg
}  // namespace gert
