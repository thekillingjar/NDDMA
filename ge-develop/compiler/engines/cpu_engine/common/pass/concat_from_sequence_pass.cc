/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "concat_from_sequence_pass.h"
#include "graph_metadef/graph/debug/ge_util.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/debug/log.h"
#include "framework/common/ge_inner_error_codes.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/type_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "common/util/mem_utils.h"
#include "util/log.h"

using namespace ge;

namespace aicpu {
namespace {
static const std::string OP_CONSTANT = "Const";
static const std::string OP_ADD = "Add";
static const std::string OP_SEQUENCEAT = "SequenceAt";
static const std::string OP_CONCAT = "Concat";
static const std::string OP_UNSQUEEZE = "Unsqueeze";
static const std::string OP_SEQUENCELENGTH = "SequenceLength";
static const std::string OP_LESS = "Less";
static const std::string OP_IDENTITY = "Identity";
static const std::string OP_WHILE = "While";
static const std::string OP_CONCATFROMSEQUENCE = "ConcatFromSequence";

static std::atomic_long const_num(0);
static std::atomic_long sequence_at_num(0);
static std::atomic_long add_num(0);
static std::atomic_long unsqueeze_num(0);
static std::atomic_long concat_num(0);
static std::atomic_long sequence_length_num(0);
static std::atomic_long less_num(0);
static std::atomic_long while_num(0);
static std::atomic_long cond_num(0);
static std::atomic_long body_num(0);
static std::atomic_long identity_num(0);

const uint32_t KWhileInputVariableIndex = 0;
const uint32_t kWhileHandleIndex = 1;
const uint32_t KWhileSequenceAtIndex = 2;
const uint32_t KWhileAddIndex = 3;
}  // namespace

OpDescPtr CreateConstDesc(const std::string &name, int32_t value) {
  OpDescPtr const_op_desc = MakeShared<OpDesc>(name, OP_CONSTANT);
  if (const_op_desc == nullptr) {
    AICPUE_LOGE("[New][OpDesc] failed.");
    return nullptr;
  }

  GeTensorDesc data_desc(GeShape(), FORMAT_ND, DT_INT32);
  GeTensorPtr const_value = MakeShared<GeTensor>(data_desc, reinterpret_cast<uint8_t *>(&value), sizeof(int32_t));
  if (const_value == nullptr) {
    AICPUE_LOGE("[New][GeTensor] failed");
    return nullptr;
  }

  if (!AttrUtils::SetTensor(const_op_desc, ATTR_NAME_WEIGHTS, const_value)) {
    AICPUE_LOGE("[Set][Attr] %s to op:%s(%s) failed", ATTR_NAME_WEIGHTS.c_str(), const_op_desc->GetName().c_str(),
                const_op_desc->GetType().c_str());
    return nullptr;
  }

  if (const_op_desc->AddOutputDesc("y", data_desc) != GRAPH_SUCCESS) {
    AICPUE_LOGE("[Add][OutputDesc] to op:%s(%s) failed, name:y", const_op_desc->GetName().c_str(),
                const_op_desc->GetType().c_str());
    return nullptr;
  }

  return const_op_desc;
}

void SetUnknowRankAndDataType(GeTensorDescPtr &tensor_desc, ge::DataType data_type) {
  const GeShape temp_shape(ge::UNKNOWN_RANK);
  tensor_desc->SetShape(temp_shape);
  tensor_desc->SetOriginShape(temp_shape);
  tensor_desc->SetDataType(data_type);
  tensor_desc->SetOriginDataType(data_type);
}

OpDescPtr CreateSequenceAtDesc(const std::string &sequence_at_name, ge::DataType data_type) {
  OpDescBuilder sequence_at_op_desc_builder(sequence_at_name, OP_SEQUENCEAT);
  OpDescPtr sequence_at_op_desc =
      sequence_at_op_desc_builder.AddInput("handle").AddInput("index").AddOutput("y").Build();
  if (sequence_at_op_desc == nullptr) {
    AICPUE_LOGE("[New][OpDesc] sequence_at_op_desc is nullptr.");
    return nullptr;
  }
  auto sequence_at_in_desc0 = sequence_at_op_desc->MutableInputDesc(0);
  sequence_at_in_desc0->SetDataType(DT_RESOURCE);
  sequence_at_in_desc0->SetOriginDataType(DT_RESOURCE);

  auto sequence_at_in_desc1 = sequence_at_op_desc->MutableInputDesc(1);
  sequence_at_in_desc1->SetDataType(DT_INT32);
  sequence_at_in_desc1->SetOriginDataType(DT_INT32);

  auto sequence_at_output_desc = sequence_at_op_desc->MutableOutputDesc(0);
  SetUnknowRankAndDataType(sequence_at_output_desc, data_type);
  return sequence_at_op_desc;
}

OpDescPtr CreateUnsqueezeDesc(const std::string &unsqueeze_name, ge::DataType data_type, int64_t axis) {
  OpDescBuilder unsqueeze_op_desc_builder(unsqueeze_name, OP_UNSQUEEZE);
  OpDescPtr unsqueeze_op_desc = unsqueeze_op_desc_builder.AddInput("x").AddOutput("y").Build();
  if (unsqueeze_op_desc == nullptr) {
    AICPUE_LOGE("[New][OpDesc] unsqueeze_op_desc is nullptr.");
    return nullptr;
  }
  std::vector<int64_t> axes = {axis};
  AttrUtils::SetListInt(unsqueeze_op_desc, "axes", axes);
  auto unsqueeze_input_desc = unsqueeze_op_desc->MutableInputDesc(0);
  SetUnknowRankAndDataType(unsqueeze_input_desc, data_type);
  auto unsqueeze_output_desc = unsqueeze_op_desc->MutableOutputDesc(0);
  SetUnknowRankAndDataType(unsqueeze_output_desc, data_type);
  return unsqueeze_op_desc;
}

OpDescPtr CreateWhileDesc() {
  auto while_next_num = while_num.fetch_add(1);
  std::string while_name = "While_ConcatFromSequencePass_" + std::to_string(while_next_num);
  OpDescBuilder while_op_desc_builder(while_name, OP_WHILE);
  // while 节点要求input num与output num相等
  OpDescPtr while_op_desc = while_op_desc_builder.AddInput("x1")
                                .AddInput("handle")
                                .AddInput("x2")
                                .AddOutput("y1")
                                .AddOutput("y2")
                                .AddOutput("y3")
                                .Build();
  if (while_op_desc == nullptr) {
    AICPUE_LOGE("[New][OpDesc] while_op_desc is nullptr.");
    return nullptr;
  }
  return while_op_desc;
}

ComputeGraphPtr BuildBodyGraph(const NodePtr &while_node, int32_t axis, int32_t new_axis,
                               const DataType output_data_type) {
  std::string body_name = "Body_ConcatFromSequencePass_" + std::to_string(body_num.fetch_add(1));
  CompleteGraphBuilder graph_builder(body_name, false);
  graph_builder.SetParentNode(while_node);

  auto sequence_at_next_num = sequence_at_num.fetch_add(1);
  std::string sequence_at_name = "Sequence_At_ConcatFromSequencePass_Body_" + std::to_string(sequence_at_next_num);
  auto sequence_at_op_desc = CreateSequenceAtDesc(sequence_at_name, output_data_type);
  graph_builder.AddNode(sequence_at_op_desc);

  std::string const_axis = "Const_Axis_ConcatFromSequencePass_Body_" + std::to_string(const_num.fetch_add(1));
  OpDescPtr const_axis_desc = CreateConstDesc(const_axis, axis);
  graph_builder.AddNode(const_axis_desc);

  auto concat_next_num = concat_num.fetch_add(1);
  std::string concat_name = "Concat_ConcatFromSequencePass_Body_" + std::to_string(concat_next_num);
  OpDescBuilder concat_op_desc_builder(concat_name, OP_CONCAT);
  OpDescPtr concat_op_desc =
      concat_op_desc_builder.AddInput("concat_dim").AddInput("x0").AddInput("x1").AddOutput("y").Build();
  // 2 表示每次只需要将两个tensor进行concat
  AttrUtils::SetInt(concat_op_desc, "N", 2);
  auto concat_in_desc0 = concat_op_desc->MutableInputDesc(0);
  concat_in_desc0->SetDataType(DT_INT32);
  concat_in_desc0->SetOriginDataType(DT_INT32);

  auto concat_input1_desc = concat_op_desc->MutableInputDesc(1);
  SetUnknowRankAndDataType(concat_input1_desc, output_data_type);

  auto concat_input2_desc = concat_op_desc->MutableInputDesc(2);  // 2 is the sencond input
  SetUnknowRankAndDataType(concat_input2_desc, output_data_type);

  // 将output shape设置为-2，不然后面不会对其进行infershape
  auto concat_output_desc = concat_op_desc->MutableOutputDesc(0);
  SetUnknowRankAndDataType(concat_output_desc, output_data_type);
  graph_builder.AddNode(concat_op_desc);
  graph_builder.AddDataLink(const_axis, 0, concat_name, 0);

  if (new_axis == 1) {
    auto unsequeeze_next_num = unsqueeze_num.fetch_add(1);
    std::string unsequeeze_name = "Unsqueeze_ConcatFromSequencePass_Body_" + std::to_string(unsequeeze_next_num);
    auto unsqueeze_op_desc = CreateUnsqueezeDesc(unsequeeze_name, output_data_type, axis);
    graph_builder.AddNode(unsqueeze_op_desc);
    // 2 is the second input for the concat op
    graph_builder.AddDataLink(sequence_at_name, 0, unsequeeze_name, 0).AddDataLink(unsequeeze_name, 0, concat_name, 2);
  } else {
    graph_builder.AddDataLink(sequence_at_name, 0, concat_name, 2);  // 2 is the second input for the concat op
  }

  // 对while的循环条件i进行++
  auto add_next_num = add_num.fetch_add(1);
  std::string add_name = "Add_ConcatFromSequencePass_Body_" + std::to_string(add_next_num);
  OpDescBuilder add_op_desc_build(add_name, OP_ADD);
  OpDescPtr add_op_desc = add_op_desc_build.AddInput("x1").AddInput("x2").AddOutput("y").Build();
  graph_builder.AddNode(add_op_desc);

  std::string const_add_val = "Const_Add_ConcatFromSequencePass_Body_" + std::to_string(const_num.fetch_add(1));
  OpDescPtr const_add_val_desc = CreateConstDesc(const_add_val, 1);
  graph_builder.AddNode(const_add_val_desc);
  graph_builder.AddDataLink(const_add_val, 0, add_name, 0);

  graph_builder.SetInput(KWhileInputVariableIndex, {sequence_at_name}, {1})
      .SetInput(kWhileHandleIndex, {sequence_at_name}, {0})
      .SetInput(KWhileSequenceAtIndex, {concat_name}, {1})
      .SetInput(KWhileAddIndex, {add_name}, {1});

  graph_builder.AddOutput(add_name, 0).AddOutput("Data_1", 0).AddOutput(concat_name, 0);

  uint32_t input_num = 4;
  std::map<uint32_t, uint32_t> input_mapping;
  std::map<uint32_t, uint32_t> output_mapping;
  // while节点的第一个input, 作为sequence at 的第2个input
  // while节点的第二个input，作为sequence at的第1个input
  // while 节点的第三个input，作为concat的第3个input
  // while 节点的第一个input，作为Add节点的第2个input
  for (size_t i = 0; i < input_num - 1; i++) {
    input_mapping[i] = i;
    output_mapping[i] = i;
  }
  input_mapping[input_num - 1] = 0;
  graph_builder.SetInputMapping(input_mapping);
  graph_builder.SetOutputMapping(output_mapping);

  // while 节点进行一次循环之后，会将output置为input
  graphStatus error_code = GRAPH_SUCCESS;
  std::string error_msg;
  ComputeGraphPtr body_graph = graph_builder.Build(error_code, error_msg);
  if (body_graph == nullptr) {
    AICPUE_LOGE("[Build][BodyGraph] failed: error_code:%u, error_msg:%s.", error_code, error_msg.c_str());
    return nullptr;
  }

  while_node->GetOpDesc()->AddSubgraphName(ATTR_NAME_WHILE_BODY);
  while_node->GetOpDesc()->SetSubgraphInstanceName(1, body_name);
  GE_DUMP(body_graph, "ConcatFromSequencePass_BuildBodySubGraph");
  return body_graph;
}

ComputeGraphPtr BuildCondGraph(const NodePtr &while_node) {
  auto next_cond_num = cond_num.fetch_add(1);
  std::string cond_name = "Cond_ConcatFromSequencePass_Cond_" + std::to_string(next_cond_num);
  CompleteGraphBuilder graph_builder(cond_name, false);
  graph_builder.SetParentNode(while_node);

  auto next_sequence_length_num = sequence_length_num.fetch_add(1);
  std::string sequence_length_name =
      "Sequence_Length_ConcatFromSequencePass_Cond_" + std::to_string(next_sequence_length_num);
  OpDescBuilder sequence_len_op_desc_builder(sequence_length_name, OP_SEQUENCELENGTH);
  OpDescPtr sequence_length_op_desc = sequence_len_op_desc_builder.AddInput("handle").AddOutput("length").Build();
  auto sequence_length_in_desc0 = sequence_length_op_desc->MutableInputDesc(0);
  sequence_length_in_desc0->SetDataType(DT_RESOURCE);
  sequence_length_in_desc0->SetOriginDataType(DT_RESOURCE);

  auto sequence_length_in_desc1 = sequence_length_op_desc->MutableOutputDesc(0);
  sequence_length_in_desc1->SetDataType(DT_INT64);
  sequence_length_in_desc1->SetOriginDataType(DT_INT64);
  graph_builder.AddNode(sequence_length_op_desc);

  auto less_next_num = less_num.fetch_add(1);
  std::string less_name = "Less_ConcatFromSequencePass_Cond_" + std::to_string(less_next_num);
  OpDescBuilder less_op_desc_builder(less_name, OP_LESS);
  OpDescPtr less_op_desc = less_op_desc_builder.AddInput("x1").AddInput("x2").AddOutput("y").Build();
  auto less_out_desc0 = less_op_desc->MutableOutputDesc(0);
  less_out_desc0->SetDataType(DT_BOOL);
  less_out_desc0->SetOriginDataType(DT_BOOL);
  graph_builder.AddNode(less_op_desc);

  graph_builder.SetInput(KWhileInputVariableIndex, {less_name}, {0})
      .SetInput(kWhileHandleIndex, {sequence_length_name}, {0})
      .SetUselessInput(KWhileSequenceAtIndex);
  graph_builder.AddOutput(less_name, 0);
  graph_builder.AddDataLink(sequence_length_name, 0, less_name, 1);

  uint32_t input_num = 3;
  // 设置while 节点的input与cond graph 的input的映射关系
  // 这里有一个孤立的data 节点
  // cond 节点的output不需要与while节点进行映射
  std::map<uint32_t, uint32_t> input_mapping;
  for (size_t i = 0; i < input_num; i++) {
    input_mapping[i] = i;
  }
  graph_builder.SetInputMapping(input_mapping);
  graphStatus error_code = GRAPH_SUCCESS;
  std::string error_msg;
  ComputeGraphPtr cond_graph = graph_builder.Build(error_code, error_msg);
  if (cond_graph == nullptr) {
    AICPUE_LOGE("[Build][CondGraph] failed: error_code:%u, error_msg:%s.", error_code, error_msg.c_str());
    return nullptr;
  }
  GE_DUMP(cond_graph, "ConcatFromSequencePass_condSubGraph");
  while_node->GetOpDesc()->AddSubgraphName(ATTR_NAME_WHILE_COND);
  while_node->GetOpDesc()->SetSubgraphInstanceName(0, cond_name);
  return cond_graph;
}

uint32_t AddCondAndBodyGraph(ComputeGraphPtr &graph, NodePtr &while_node, const NodePtr &concat_from_sequence_node) {
  ComputeGraphPtr root_graph = GraphUtils::FindRootGraph(graph);
  GE_CHECK_NOTNULL(root_graph);

  // 通过const节点生产identity节点作为while节点的第一个input
  // concatFromSequence node 原本input，即handle作为while节点的第二个input
  // sequence at 作为while node 的第三个input
  ComputeGraphPtr cond_graph = BuildCondGraph(while_node);
  if ((cond_graph == nullptr) || (root_graph->AddSubgraph(cond_graph)) != GRAPH_SUCCESS) {
    AICPUE_LOGE("Root graph add cond graph failed");
    return FAILED;
  }
  int32_t axis = 0;
  AttrUtils::GetInt(concat_from_sequence_node->GetOpDesc(), "axis", axis);
  int32_t new_axis = 0;
  AttrUtils::GetInt(concat_from_sequence_node->GetOpDesc(), "new_axis", new_axis);
  auto concat_from_sequence_desc = concat_from_sequence_node->GetOpDesc();
  auto concat_from_sequence_output_data_type = concat_from_sequence_desc->GetOutputDesc(0).GetDataType();
  // 将input2 shape设置为-2，不然后面承接的cast算子不会走infershape
  auto while_op_desc = while_node->GetOpDesc();
  auto while_input2_desc = while_op_desc->MutableInputDesc(2);  // 2 is the sencond input
  SetUnknowRankAndDataType(while_input2_desc, concat_from_sequence_output_data_type);
  // 将output2 shape设置为-2，不然后面承接的cast算子不会走infershape
  auto while_output2_desc = while_op_desc->MutableOutputDesc(2);  // 2 is the sencond output
  SetUnknowRankAndDataType(while_output2_desc, concat_from_sequence_output_data_type);
  ComputeGraphPtr body_graph = BuildBodyGraph(while_node, axis, new_axis, concat_from_sequence_output_data_type);
  if ((body_graph == nullptr) || (root_graph->AddSubgraph(body_graph)) != GRAPH_SUCCESS) {
    AICPUE_LOGE("Root graph add body graph failed");
    return FAILED;
  }

  NodeUtils::UnlinkAll(*concat_from_sequence_node);
  graph->RemoveNode(concat_from_sequence_node);
  return SUCCESS;
}

uint32_t BuildWhileSubGraph(ComputeGraphPtr &graph, const NodePtr &concat_from_sequence_node) {
  GE_DUMP(graph, "before_ConcatFromSequencePass");
  auto identity_next_num = identity_num.fetch_add(1);
  std::string identity_name = "Identity_concatFromSeuqencePass_" + std::to_string(identity_next_num);
  OpDescBuilder identity_op_desc_builder(identity_name, OP_IDENTITY);
  OpDescPtr identity_op_desc = identity_op_desc_builder.AddInput("x").AddOutput("y").Build();
  GE_CHECK_NOTNULL(identity_op_desc);
  // Maybe we need to Repass the new node
  NodePtr identity_node = graph->AddNode(identity_op_desc);

  std::string const_name = "Const_ConcatFromSequencePass_" + std::to_string(const_num.fetch_add(1));
  OpDescPtr const_op_desc = CreateConstDesc(const_name, 0);
  GE_CHECK_NOTNULL(const_op_desc);
  NodePtr const_node = graph->AddNode(const_op_desc);

  auto concat_from_sequence_desc = concat_from_sequence_node->GetOpDesc();
  auto concat_from_sequence_output_data_type = concat_from_sequence_desc->GetOutputDesc(0).GetDataType();

  auto sequence_at_next_num = sequence_at_num.fetch_add(1);
  std::string sequence_at_name = "Sequence_At_ConcatFromSequencePass_" + std::to_string(sequence_at_next_num);
  auto sequence_at_op_desc = CreateSequenceAtDesc(sequence_at_name, concat_from_sequence_output_data_type);
  GE_CHECK_NOTNULL(sequence_at_op_desc);
  NodePtr sequence_at_node = graph->AddNode(sequence_at_op_desc);
  GE_CHECK_NOTNULL(const_node);
  GE_CHECK_NOTNULL(sequence_at_node);
  if (GraphUtils::AddEdge(const_node->GetOutDataAnchor(0), sequence_at_node->GetInDataAnchor(1)) != GRAPH_SUCCESS) {
    AICPUE_LOGE("[Build][While]Add edge between const_node and sequence_at_node failed");
    return FAILED;
  }

  std::string const_1_name = "Const_ConcatFromSequencePass_" + std::to_string(const_num.fetch_add(1));
  // while body体内从sequence tensor 的第二个元素开始
  OpDescPtr const_1_op_desc = CreateConstDesc(const_1_name, 1);
  GE_CHECK_NOTNULL(const_1_op_desc);
  NodePtr const_1_node = graph->AddNode(const_1_op_desc);
  GE_CHECK_NOTNULL(const_1_node);
  // 因为需要修改这个const节点的值，所以连接一个identity node
  if (GraphUtils::AddEdge(const_1_node->GetOutDataAnchor(0), identity_node->GetInDataAnchor(0)) != GRAPH_SUCCESS) {
    AICPUE_LOGE("[Build][While]Add edge between const_1_node and identity_node failed");
    return FAILED;
  }

  InDataAnchorPtr cur_node_in_data_anchor = concat_from_sequence_node->GetInDataAnchor(0);
  OutDataAnchorPtr cur_node_peer_out_anchor = cur_node_in_data_anchor->GetPeerOutAnchor();
  if (GraphUtils::AddEdge(cur_node_peer_out_anchor, sequence_at_node->GetInDataAnchor(0)) != GRAPH_SUCCESS) {
    AICPUE_LOGE("[Build][While]Add edge between cur_node_peer_out_anchor and sequence_at_node[0] failed");
    return FAILED;
  }
  auto while_op_desc = CreateWhileDesc();
  GE_CHECK_NOTNULL(while_op_desc);
  NodePtr while_node = graph->AddNode(while_op_desc);
  GE_CHECK_NOTNULL(while_node);

  int32_t axis = 0;
  AttrUtils::GetInt(concat_from_sequence_node->GetOpDesc(), "axis", axis);
  int32_t new_axis = 0;
  AttrUtils::GetInt(concat_from_sequence_node->GetOpDesc(), "new_axis", new_axis);
  if (new_axis == 0) {
    // 当while的body体进不去时，while节点的input会作为output
    // 2 is the sencond input of while node
    if (GraphUtils::AddEdge(sequence_at_node->GetOutDataAnchor(0), while_node->GetInDataAnchor(2)) != GRAPH_SUCCESS) {
      AICPUE_LOGE("[Build][While]Add edge between sequence_at_node and while_node[2] failed");
      return FAILED;
    }
  } else {
    auto unsqueeze_next_num = unsqueeze_num.fetch_add(1);
    std::string unsqueeze_name = "Unsqueeze_ConcatFromSequencePass_" + std::to_string(unsqueeze_next_num);
    auto unsqueeze_op_desc = CreateUnsqueezeDesc(unsqueeze_name, concat_from_sequence_output_data_type, axis);
    GE_CHECK_NOTNULL(unsqueeze_op_desc);
    NodePtr unsqueeze_node = graph->AddNode(unsqueeze_op_desc);
    GE_CHECK_NOTNULL(unsqueeze_node);

    if (GraphUtils::AddEdge(sequence_at_node->GetOutDataAnchor(0), unsqueeze_node->GetInDataAnchor(0)) !=
        GRAPH_SUCCESS) {
      AICPUE_LOGE("[Build][While]Add edge between sequence_at_node and unsqueeze_node[0] failed");
      return FAILED;
    }

    // 2 is the sencond input of while node
    if (GraphUtils::AddEdge(unsqueeze_node->GetOutDataAnchor(0), while_node->GetInDataAnchor(2)) != GRAPH_SUCCESS) {
      AICPUE_LOGE("[Build][While]Add edge between unsqueeze_node and while_node[2] failed");
      return FAILED;
    }
  }

  if (GraphUtils::AddEdge(identity_node->GetOutDataAnchor(0), while_node->GetInDataAnchor(0)) != GRAPH_SUCCESS) {
    AICPUE_LOGE("[Build][While]Add edge between identity_node and while_node[0] failed");
    return FAILED;
  };

  InDataAnchorPtr in_data_anchor = while_node->GetInDataAnchor(1);
  // concatFromSequence 原本input 即handle 作为while节点的第二个input
  if (GraphUtils::AddEdge(cur_node_peer_out_anchor, in_data_anchor) != GRAPH_SUCCESS) {
    AICPUE_LOGE("[Build][While]Add edge between cur_node_peer_out_anchor and in_data_anchor failed");
    return FAILED;
  };

  OutDataAnchorPtr while_out_data_anchor = while_node->GetOutDataAnchor(2);  // 2 is the sencond output
  OutDataAnchorPtr cur_node_out_data_anchor = concat_from_sequence_node->GetOutDataAnchor(0);
  // 将原本连接concatFromSequence output的节点与while node的第二个output进行连接
  for (const auto &peer_in_data_anchor : cur_node_out_data_anchor->GetPeerInDataAnchors()) {
    if (peer_in_data_anchor == nullptr) {
      AICPUE_LOGE("cur_node_out_data_anchor's peer_in_data_anchor is nullptr");
      return FAILED;
    }
    if (GraphUtils::RemoveEdge(cur_node_out_data_anchor, peer_in_data_anchor) != GRAPH_SUCCESS) {
      AICPUE_LOGE("Remove edge between cur_node_out_data_anchor and peer_in_data_anchor failed");
      return FAILED;
    }
    if (GraphUtils::AddEdge(while_out_data_anchor, peer_in_data_anchor) != GRAPH_SUCCESS) {
      AICPUE_LOGE("Add edge between cur_node_out_data_anchor and peer_in_data_anchor failed");
      return FAILED;
    }
  }
  if (AddCondAndBodyGraph(graph, while_node, concat_from_sequence_node) != SUCCESS) {
    AICPUE_LOGE("AddCondAndBodyGraph failed");
    return FAILED;
  }
  GE_DUMP(graph, "after_ConcatFromSequencePass");
  return SUCCESS;
}

Status ConcatFromSequencePass::Run(ge::ComputeGraph &graph) {
  for (const auto &node : graph.GetAllNodes()) {
    if (OP_CONCATFROMSEQUENCE == node->GetType()) {
      ComputeGraphPtr owner_compute_graph = node->GetOwnerComputeGraph();
      GE_CHECK_NOTNULL(owner_compute_graph);
      if (BuildWhileSubGraph(owner_compute_graph, node) != SUCCESS) {
        AICPUE_LOGE("[ConcatFromSequencePass] failed to create while graph");
        return FAILED;
      }
    }
  }
  return SUCCESS;
}
}  // namespace aicpu