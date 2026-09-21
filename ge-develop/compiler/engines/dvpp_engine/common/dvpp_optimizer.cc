/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dvpp_optimizer.h"
#include "common/dvpp_ops_checker.h"
#include "common/dvpp_ops_lib.h"
#include "util/dvpp_constexpr.h"
#include "util/dvpp_define.h"
#include "util/dvpp_error_code.h"
#include "util/dvpp_log.h"
#include "util/util.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"

namespace dvpp {
bool DvppOptimizer::EnableDvppEngine(ge::OpDescPtr& op_desc_ptr) const {
  // 检查属性 ATTR_NAME_PERFORMANCE_PRIOR 确认是否使能dvpp engine
  bool performance_prior = false;
  bool ret = ge::AttrUtils::GetBool(
      op_desc_ptr, ATTR_NAME_PERFORMANCE_PRIOR, performance_prior);
  DVPP_CHECK_IF_THEN_DO(!ret, return false);

  return performance_prior;
}

bool DvppOptimizer::CheckSupportedSingleOpGraph(ge::ComputeGraph& graph)
    const {
  for (auto& node : graph.GetDirectNode()) {
    // 判断依据无视Data Const Constant NetOutput节点
    DVPP_CHECK_IF_THEN_DO((node->GetType() == kNodeData) ||
        (node->GetType() == kNodeNetOutput) ||
        (node->GetType() == kNodeConst) || (node->GetType() == kNodeConstant),
        continue);

    // check enable dvpp engine
    auto op_desc_ptr = node->GetOpDesc();
    bool enable_dvpp_engine = EnableDvppEngine(op_desc_ptr);
    DVPP_CHECK_IF_THEN_DO(!enable_dvpp_engine, return false);

    DVPP_ENGINE_LOG_DEBUG("dvpp engine is enable to op[%s].",
                          op_desc_ptr->GetType().c_str());

    // check support
    std::string unsupported_reason;
    return DvppOpsChecker::CheckSupported(node, unsupported_reason);
  }

  return false;
}

/*
在Pytorch单算子场景下，把所有data节点输入、输出shape设置为动态shape
1）常规算子，将data节点的根据dim num的设置shape为dim num个 -1
2）特殊算子(Decode算子)，DT_STRING类型，将data节点shape设置为 {-2}
*/
DvppErrorCode DvppOptimizer::SetGraphDynamicShape(ge::ComputeGraph& graph) {
  DVPP_ENGINE_LOG_DEBUG("dvpp optimizer SetGraphDynamicShape start.");
  // 配置data节点输入、输出shape
  for (auto& node : graph.GetDirectNode()) {
    DVPP_CHECK_IF_THEN_DO(node->GetType() != kNodeData, continue);

    auto op_desc_ptr = node->GetOpDesc();
    DVPP_CHECK_NOTNULL(op_desc_ptr);

    auto input = op_desc_ptr->MutableInputDesc(0);
    DVPP_CHECK_NOTNULL(input);

    auto output = op_desc_ptr->MutableOutputDesc(0);
    DVPP_CHECK_NOTNULL(output);

    ge::GeShape dynamic_shape;
    if (input->GetDataType() == ge::DT_STRING) {
      dynamic_shape = ge::GeShape({kDynamicShape});
    } else {
      std::vector<int64_t> dims = input->GetShape().GetDims();
      dynamic_shape = ge::GeShape(std::vector<int64_t>(dims.size(),
          kDynamicDim));
      std::vector<std::pair<int64_t, int64_t>> range;
      GetShapeRangeByDataFormat(range, dims, input->GetFormat());
      input->SetShapeRange(range);
      input->SetOriginShapeRange(range);
      output->SetShapeRange(range);
      output->SetOriginShapeRange(range);
    }

    input->SetOriginShape(ge::GeShape(dynamic_shape));
    input->SetShape(ge::GeShape(dynamic_shape));
    output->SetOriginShape(ge::GeShape(dynamic_shape));
    output->SetShape(ge::GeShape(dynamic_shape));
  }

  DVPP_ENGINE_LOG_DEBUG("dvpp optimizer SetGraphDynamicShape success.");
  return DvppErrorCode::kSuccess;
}

DvppErrorCode DvppOptimizer::OptimizeGraphPrepare(ge::ComputeGraph& graph) {
  // 当前主要用于在pytorch单算子场景下 设置动态shape
  // 因为时机必须在infer shape之前 所以在此函数设置
  DVPP_ENGINE_LOG_DEBUG("dvpp optimizer OptimizeGraphPrepare start.");
  bool is_single_op = false;
  (void)ge::AttrUtils::GetBool(graph, ge::ATTR_SINGLE_OP_SCENE, is_single_op);
  DVPP_CHECK_IF_THEN_DO(!is_single_op, return DvppErrorCode::kSuccess);

  DVPP_ENGINE_LOG_DEBUG("this graph is single op graph.");

  bool is_support = CheckSupportedSingleOpGraph(graph);
  DVPP_CHECK_IF_THEN_DO(!is_support, return DvppErrorCode::kSuccess);

  DVPP_ENGINE_LOG_DEBUG("dvpp support this single op graph.");

  auto error = SetGraphDynamicShape(graph);
  DVPP_CHECK_IF_THEN_DO(error != DvppErrorCode::kSuccess,
      DVPP_REPORT_INNER_ERR_MSG("set graph dynamic shape failed");
      return error);
  DVPP_ENGINE_LOG_DEBUG("dvpp optimizer OptimizeGraphPrepare success.");
  return DvppErrorCode::kSuccess;
}

// ge:Maximum  ge:Constant           ge:Maximum  ge:Constant
//        \     /                           \      /  ge:Constant
//          Sub  ge:Constant   ===>          \    /     /
//            \   /                           NormalizeV2
//             Mul
DvppErrorCode DvppOptimizer::SubAndMulFusedIntoNormalizeV2(
    ge::ComputeGraph& graph, ge::NodePtr& mul_node) const {
  // 先获取所有关键node
  auto sub_node = mul_node->GetInDataNodes().at(kNum0);
  auto const_2_node = mul_node->GetInDataNodes().at(kNum1);
  auto const_1_node = sub_node->GetInDataNodes().at(kNum1);

  // 在Sub和Mul之间插入NormalizeV2
  ge::ComputeGraphPtr graph_ptr = graph.shared_from_this();
  ge::NodePtr normalize_v2_node = InsertGraphNodeBefore(
      graph_ptr, mul_node, kDvppOpNormalizeV2);
  DVPP_CHECK_IF_THEN_DO(normalize_v2_node == nullptr,
      DVPP_REPORT_INNER_ERR_MSG("insert NormalizeV2 node failed");
      return DvppErrorCode::kFailed);

  // 打上 dvpp 标签
  auto op_desc = normalize_v2_node->GetOpDesc();
  auto ret = SetDvppOpAttr(op_desc);
  DVPP_CHECK_IF_THEN_DO(ret != DvppErrorCode::kSuccess,
      DVPP_REPORT_INNER_ERR_MSG("NormalizeV2 set dvpp op attr failed");
      return DvppErrorCode::kFailed);

  // 将const1节点和NormalizeV2连边
  auto const_1_anchor = const_1_node->GetOutDataAnchor(kNum0);
  auto normalize_v2_anchor = normalize_v2_node->GetInDataAnchor(kNum1);
  auto error = ge::GraphUtils::AddEdge(const_1_anchor, normalize_v2_anchor);
  DVPP_CHECK_IF_THEN_DO(error != ge::GRAPH_SUCCESS,
      DVPP_REPORT_INNER_ERR_MSG("add edge Const1 and NormalizeV2 failed,%d.",
                              error);
      return DvppErrorCode::kFailed);

  // 将const2节点和NormalizeV2连边
  auto const_2_anchor = const_2_node->GetOutDataAnchor(kNum0);
  normalize_v2_anchor = normalize_v2_node->GetInDataAnchor(kNum2);
  error = ge::GraphUtils::AddEdge(const_2_anchor, normalize_v2_anchor);
  DVPP_CHECK_IF_THEN_DO(error != ge::GRAPH_SUCCESS,
      DVPP_REPORT_INNER_ERR_MSG("add edge Const2 and NormalizeV2 failed, %d",
                              error);
      return DvppErrorCode::kFailed);

  // 删除Sub
  error = graph.RemoveNode(sub_node);
  DVPP_CHECK_IF_THEN_DO(error != ge::GRAPH_SUCCESS,
      DVPP_REPORT_INNER_ERR_MSG("remove Sub node failed, error:%d", error);
      return DvppErrorCode::kFailed);

  // 删除Mul
  error = graph.RemoveNode(mul_node);
  DVPP_CHECK_IF_THEN_DO(error != ge::GRAPH_SUCCESS,
      DVPP_REPORT_INNER_ERR_MSG("remove Mul node failed, error:%d", error);
      return DvppErrorCode::kFailed);

  return DvppErrorCode::kSuccess;
}

DvppErrorCode DvppOptimizer::SpecialOptimize(
    ge::ComputeGraph& graph, ge::NodePtr& node) const {
  // 当前仅支持特定场景下的Sub和Mul算子融合为Normalize算子
  bool ret = DvppOpsChecker::CheckSubAndMulFusedIntoNormalizeV2(node);
  DVPP_CHECK_IF_THEN_DO(!ret, return DvppErrorCode::kFailed);

  DVPP_ENGINE_LOG_DEBUG("dvpp support Sub and Mul fused into NormalizeV2");

  // do fused
  auto error = SubAndMulFusedIntoNormalizeV2(graph, node);
  DVPP_CHECK_IF_THEN_DO(error != DvppErrorCode::kSuccess, return error);

  DVPP_ENGINE_LOG_DEBUG("Sub and Mul fused into NormalizeV2 success");

  return DvppErrorCode::kSuccess;
}

DvppErrorCode DvppOptimizer::SetMemoryType(ge::OpDescPtr& op_desc_ptr) const {
  // 设置 ATTR_NAME_INPUT_MEM_TYPE_LIST & ATTR_NAME_OUTPUT_MEM_TYPE_LIST
  std::string op_type = op_desc_ptr->GetType();
  DvppOpInfo dvpp_op_info;
  bool ret = DvppOpsLib::Instance().GetDvppOpInfo(op_type, dvpp_op_info);
  DVPP_CHECK_IF_THEN_DO(!ret,
      DVPP_REPORT_INNER_ERR_MSG("can not find op[%s] in dvpp info store.",
                              op_type.c_str());
      return DvppErrorCode::kFailed);

  // td 当前没有定义RT_MEMORY_DVPP 这个需要对齐一下
  // 如果是默认 RT_MEMORY_HBM (0x2U) 则不打标签
  int64_t memory_type = 2; // dvpp_op_info.memoryType
  DVPP_CHECK_IF_THEN_DO(memory_type == 2, return DvppErrorCode::kSuccess);

  // set input memory type
  size_t inputs_size = op_desc_ptr->GetInputsSize();
  std::vector<int64_t> inputs_memory_type(inputs_size, memory_type);
  ret = ge::AttrUtils::SetListInt(
      op_desc_ptr, ge::ATTR_NAME_INPUT_MEM_TYPE_LIST, inputs_memory_type);
  DVPP_CHECK_IF_THEN_DO(!ret,
      DVPP_REPORT_INNER_ERR_MSG("op[%s] set input memory type failed",
                              op_type.c_str());
      return DvppErrorCode::kFailed);

  // set output memory type
  size_t outputs_size = op_desc_ptr->GetOutputsSize();
  std::vector<int64_t> outputs_memory_type(outputs_size, memory_type);
  ret = ge::AttrUtils::SetListInt(
      op_desc_ptr, ge::ATTR_NAME_OUTPUT_MEM_TYPE_LIST, outputs_memory_type);
  DVPP_CHECK_IF_THEN_DO(!ret,
      DVPP_REPORT_INNER_ERR_MSG("op[%s] set output memory type failed",
                              op_type.c_str());
      return DvppErrorCode::kFailed);

  return DvppErrorCode::kSuccess;
}

DvppErrorCode DvppOptimizer::SetDvppOpAttr(ge::OpDescPtr& op_desc_ptr) const {
  DVPP_CHECK_NOTNULL(op_desc_ptr);
  // set ATTR_NAME_OP_SPECIFIED_ENGINE_NAME
  bool ret = ge::AttrUtils::SetStr(
      op_desc_ptr, ge::ATTR_NAME_OP_SPECIFIED_ENGINE_NAME, kDvppEngineName);
  DVPP_CHECK_IF_THEN_DO(!ret,
      DVPP_REPORT_INNER_ERR_MSG("op[%s] set op specified engine name failed",
                              op_desc_ptr->GetType().c_str());
      return DvppErrorCode::kFailed);

  // set ATTR_NAME_OP_SPECIFIED_KERNEL_LIB_NAME
  ret = ge::AttrUtils::SetStr(op_desc_ptr,
      ge::ATTR_NAME_OP_SPECIFIED_KERNEL_LIB_NAME, kDvppOpsKernel);
  DVPP_CHECK_IF_THEN_DO(!ret,
      DVPP_REPORT_INNER_ERR_MSG("op[%s] set op specified kernel lib name failed",
                              op_desc_ptr->GetType().c_str());
      return DvppErrorCode::kFailed);

  // set ATTR_NAME_FORCE_UNKNOWN_SHAPE 强制设置为动态算子
  ret = ge::AttrUtils::SetBool(
      op_desc_ptr, ge::ATTR_NAME_FORCE_UNKNOWN_SHAPE, true);
  DVPP_CHECK_IF_THEN_DO(!ret,
      DVPP_REPORT_INNER_ERR_MSG("op[%s] set force unknown shape failed",
                              op_desc_ptr->GetType().c_str());
      return DvppErrorCode::kFailed);

  // set memory type
  auto error = SetMemoryType(op_desc_ptr);
  DVPP_CHECK_IF_THEN_DO(error != DvppErrorCode::kSuccess,
        DVPP_REPORT_INNER_ERR_MSG("op[%s] set memory type failed",
                                op_desc_ptr->GetType().c_str());
        return error);

  return DvppErrorCode::kSuccess;
}

DvppErrorCode DvppOptimizer::SetAiCpuOpAttr(ge::OpDescPtr& op_desc_ptr) const {
  DVPP_CHECK_NOTNULL(op_desc_ptr);
  // set ATTR_NAME_OP_SPECIFIED_ENGINE_NAME
  bool ret = ge::AttrUtils::SetStr(
      op_desc_ptr, ge::ATTR_NAME_OP_SPECIFIED_ENGINE_NAME, kAiCpuEngineName);
  DVPP_CHECK_IF_THEN_DO(!ret,
      DVPP_REPORT_INNER_ERR_MSG("op[%s] set op specified engine name failed.",
                              op_desc_ptr->GetType().c_str());
      return DvppErrorCode::kFailed);

  // set ATTR_NAME_OP_SPECIFIED_KERNEL_LIB_NAME
  ret = ge::AttrUtils::SetStr(op_desc_ptr,
      ge::ATTR_NAME_OP_SPECIFIED_KERNEL_LIB_NAME, kAiCpuDecodeOpsKernel);
  DVPP_CHECK_IF_THEN_DO(!ret,
      DVPP_REPORT_INNER_ERR_MSG("op[%s] set op specified kernel lib name failed.",
                              op_desc_ptr->GetType().c_str());
      return DvppErrorCode::kFailed);

  // set ATTR_NAME_FORCE_UNKNOWN_SHAPE 强制设置为动态算子
  ret = ge::AttrUtils::SetBool(
      op_desc_ptr, ge::ATTR_NAME_FORCE_UNKNOWN_SHAPE, true);
  DVPP_CHECK_IF_THEN_DO(!ret,
      DVPP_REPORT_INNER_ERR_MSG("op[%s] set force unknown shape failed",
                              op_desc_ptr->GetType().c_str());
      return DvppErrorCode::kFailed);

  return DvppErrorCode::kSuccess;
}

DvppErrorCode DvppOptimizer::GetParentGraphSessionId(
    ge::ComputeGraphPtr graph,
    std::string& parent_graph_session_id) const {
  bool ret = ge::AttrUtils::GetStr(graph, ge::ATTR_NAME_SESSION_GRAPH_ID,
                                   parent_graph_session_id);
  DVPP_CHECK_IF_THEN_DO(!ret,
      DVPP_REPORT_INNER_ERR_MSG("get parent graph session id failed.");
      return DvppErrorCode::kFailed);

    return DvppErrorCode::kSuccess;
}

DvppErrorCode DvppOptimizer::SetSubgraphParentSessionId(
    ge::ComputeGraphPtr& then_graph, ge::ComputeGraphPtr& else_graph,
    std::string& parent_graph_session_id) const {
  bool ret = ge::AttrUtils::SetStr(then_graph, ge::ATTR_NAME_SESSION_GRAPH_ID,
                                   parent_graph_session_id);
  DVPP_CHECK_IF_THEN_DO(!ret,
      DVPP_REPORT_INNER_ERR_MSG("set then_graph parent graph session id failed.");
      return DvppErrorCode::kFailed);

  ret = ge::AttrUtils::SetStr(else_graph, ge::ATTR_NAME_SESSION_GRAPH_ID,
                              parent_graph_session_id);
  DVPP_CHECK_IF_THEN_DO(!ret,
      DVPP_REPORT_INNER_ERR_MSG("set else_graph parent graph session id failed.");
      return DvppErrorCode::kFailed);

  return DvppErrorCode::kSuccess;
}

ge::NodePtr DvppOptimizer::InsertGraphNodeBefore(ge::ComputeGraphPtr& graph,
    ge::NodePtr& node, const std::string& op_type) const {
  // create op_desc
  ge::OpDescPtr op_desc_ptr = nullptr;
  auto ret = DvppErrorCode::kFailed;
  if (op_type == kDvppOpIf) {
    ret = DvppOpsLib::Instance().CreateIfOpDesc(node->GetType(), op_desc_ptr);
  } else {
    ret = DvppOpsLib::Instance().CreateDvppOpDesc(op_type, op_desc_ptr);
  }

  DVPP_CHECK_IF_THEN_DO(ret != DvppErrorCode::kSuccess,
      DVPP_REPORT_INNER_ERR_MSG("create op[%s] failed.", op_type.c_str());
      return nullptr);

  // add node in graph
  ge::NodePtr insert_node = graph->AddNode(op_desc_ptr);
  DVPP_CHECK_IF_THEN_DO(insert_node == nullptr,
      DVPP_REPORT_INNER_ERR_MSG("create node[%s] failed.", op_type.c_str());
      return nullptr);

  // 插到node的0号输入前，且插入时insert_node输入输出位置取默认值，即0号输入和0号输出
  auto in_data_anchor = node->GetInDataAnchor(0);
  auto error = ge::GraphUtils::InsertNodeBefore(in_data_anchor, insert_node);
  DVPP_CHECK_IF_THEN_DO(error != ge::GRAPH_SUCCESS,
      DVPP_REPORT_INNER_ERR_MSG("insert node[%s] before node[%s] failed.",
                              op_type.c_str(), node->GetType().c_str());
      return nullptr);

  return insert_node;
}

DvppErrorCode DvppOptimizer::InsertNodeAfter(ge::NodePtr& node,
    ge::NodePtr& insert_node) const {
  auto out_anchor = node->GetOutDataAnchor(0);
  DVPP_CHECK_NOTNULL(out_anchor);
  auto peer_in_anchors = out_anchor->GetPeerInDataAnchors();
  std::vector<ge::InDataAnchorPtr> in_anchors(peer_in_anchors.begin(),
                                              peer_in_anchors.end());

  // 插到node的0号输出后，且插入时insert_node输入输出位置取默认值，即0号输入和0号输出
  auto error = ge::GraphUtils::InsertNodeAfter(out_anchor, in_anchors,
                                               insert_node);
  DVPP_CHECK_IF_THEN_DO(error != ge::GRAPH_SUCCESS,
      DVPP_REPORT_INNER_ERR_MSG("insert node[%s] after node[%s] failed",
                              insert_node->GetType().c_str(),
                              node->GetType().c_str());
      return DvppErrorCode::kFailed);

  return DvppErrorCode::kSuccess;
}

ge::NodePtr DvppOptimizer::InsertGraphNodeAfter(ge::ComputeGraphPtr& graph,
    ge::NodePtr& node, const std::string& op_type) const {
  auto insert_node = GraphAddNode(graph, op_type);
  DVPP_CHECK_IF_THEN_DO(insert_node == nullptr, return nullptr);

  auto ret = InsertNodeAfter(node, insert_node);
  DVPP_CHECK_IF_THEN_DO(ret != DvppErrorCode::kSuccess, return nullptr);

  return insert_node;
}

DvppErrorCode DvppOptimizer::SetSubgraphDataIndex(ge::NodePtr& data_node,
    uint32_t data_index, ge::NodePtr& output_node,
    uint32_t output_index) const {
  bool ret = ge::AttrUtils::SetInt(data_node->GetOpDesc(),
      ge::ATTR_NAME_PARENT_NODE_INDEX, data_index);
  DVPP_CHECK_IF_THEN_DO(!ret,
      DVPP_REPORT_INNER_ERR_MSG("set node[%s] subgraph index[%u] failed.",
                              data_node->GetType().c_str(), data_index);
      return DvppErrorCode::kFailed);

  ret = ge::AttrUtils::SetInt(
      output_node->GetOpDesc()->MutableInputDesc(output_index),
      ge::ATTR_NAME_PARENT_NODE_INDEX, output_index);
  DVPP_CHECK_IF_THEN_DO(!ret,
      DVPP_REPORT_INNER_ERR_MSG("set node[%s] subgraph index[%u] failed.",
                              output_node->GetType().c_str(), output_index);
      return DvppErrorCode::kFailed);

  return DvppErrorCode::kSuccess;
}

DvppErrorCode DvppOptimizer::InsertDataNodeInIfSubgraph(
    ge::ComputeGraphPtr& subgraph, ge::NodePtr& node) const {
  // create op_desc
  ge::OpDescPtr op_desc_ptr = nullptr;
  auto ret = DvppOpsLib::Instance().CreateDvppOpDesc(kNodeData, op_desc_ptr);
  DVPP_CHECK_IF_THEN_DO(ret != DvppErrorCode::kSuccess,
      DVPP_REPORT_INNER_ERR_MSG("create op[%s] failed.", kNodeData.c_str());
      return DvppErrorCode::kFailed);

  auto input_desc = op_desc_ptr->MutableInputDesc("x");
  input_desc->SetDataType(ge::DT_INT32);
  input_desc->SetShape(ge::GeShape(std::vector<int64_t>{kNum4}));
  auto output_desc = op_desc_ptr->MutableOutputDesc("y");
  output_desc->SetDataType(ge::DT_INT32);
  output_desc->SetShape(ge::GeShape(std::vector<int64_t>{kNum4}));

  // add node in graph
  ge::NodePtr insert_node = subgraph->AddNode(op_desc_ptr);
  DVPP_CHECK_IF_THEN_DO(insert_node == nullptr,
      DVPP_REPORT_INNER_ERR_MSG("create insert node[%s] failed.",
                            kNodeData.c_str());
      return DvppErrorCode::kFailed);

  // insert node
  auto insert_node_out_anchor = insert_node->GetOutDataAnchor(0);
  auto node_in_anchor = node->GetInDataAnchor(kNum1);
  auto error = ge::GraphUtils::AddEdge(insert_node_out_anchor, node_in_anchor);
  DVPP_CHECK_IF_THEN_DO(error != ge::GRAPH_SUCCESS,
      DVPP_REPORT_INNER_ERR_MSG("add edge between node[%s] and node[%s] failed.",
                              kNodeData.c_str(), node->GetType().c_str());
      return DvppErrorCode::kFailed);

  bool result = ge::AttrUtils::SetInt(insert_node->GetOpDesc(),
                                      ge::ATTR_NAME_PARENT_NODE_INDEX, kNum2);
  DVPP_CHECK_IF_THEN_DO(!result,
      DVPP_REPORT_INNER_ERR_MSG("set node[%s] subgraph index[2] failed.",
                              insert_node->GetType().c_str());
      return DvppErrorCode::kFailed);

  return DvppErrorCode::kSuccess;
}

DvppErrorCode DvppOptimizer::BuildIfThenGraph(
    ge::ComputeGraphPtr& then_graph, ge::NodePtr& node) const {
  // insert Data node
  ge::OpDescPtr data_op_desc_ptr = nullptr;
  auto ret = DvppOpsLib::Instance().CreateDvppOpDesc(kNodeData,
                                                     data_op_desc_ptr);
  DVPP_CHECK_IF_THEN_DO(ret != DvppErrorCode::kSuccess,
      DVPP_REPORT_INNER_ERR_MSG("create op[%s] failed", kNodeData.c_str());
      return DvppErrorCode::kFailed);

  ge::NodePtr data_node = then_graph->AddNode(data_op_desc_ptr);
  DVPP_CHECK_NOTNULL(data_node);

  ge::NodePtr decode_node = then_graph->AddNode(node);
  DVPP_CHECK_NOTNULL(decode_node);

  decode_node->SetOwnerComputeGraph(then_graph);

  ret = InsertNodeAfter(data_node, decode_node);
  DVPP_CHECK_IF_THEN_DO(ret != DvppErrorCode::kSuccess,
      return DvppErrorCode::kFailed);

  // 打上 dvpp 标签
  auto op_desc = decode_node->GetOpDesc();
  ret = SetDvppOpAttr(op_desc);
  DVPP_CHECK_IF_THEN_DO(ret != DvppErrorCode::kSuccess,
      DVPP_REPORT_INNER_ERR_MSG("op[%s] set dvpp op attr failed",
                              op_desc->GetType().c_str());
      return DvppErrorCode::kFailed);

  bool result = ge::AttrUtils::SetBool(op_desc, ge::NEED_INFER, true);
  DVPP_CHECK_IF_THEN_DO(!result,
      DVPP_REPORT_INNER_ERR_MSG("set op[%s] need infer attr failed.",
                              op_desc->GetType().c_str());
      return DvppErrorCode::kFailed);

  ge::NodePtr output_node = InsertGraphNodeAfter(then_graph, decode_node,
                                                 kNodeNetOutput);
  DVPP_CHECK_NOTNULL(output_node);

  ret = SetSubgraphDataIndex(data_node, 1, output_node, 0);
  DVPP_CHECK_IF_THEN_DO(ret != DvppErrorCode::kSuccess,
      DVPP_REPORT_INNER_ERR_MSG("set then subgraph data index failed.");
      return DvppErrorCode::kFailed);

  if (decode_node->GetType() == kDvppOpDecodeAndCropJpeg) {
    return InsertDataNodeInIfSubgraph(then_graph, decode_node);
  }

  return DvppErrorCode::kSuccess;
}

DvppErrorCode DvppOptimizer::BuildIfElseGraph(
    ge::ComputeGraphPtr& else_graph, ge::OpDescPtr& op_desc) const {
  // insert Data node
  ge::OpDescPtr data_op_desc_ptr = nullptr;
  auto ret = DvppOpsLib::Instance().CreateDvppOpDesc(kNodeData,
                                                     data_op_desc_ptr);
  DVPP_CHECK_IF_THEN_DO(ret != DvppErrorCode::kSuccess,
      DVPP_REPORT_INNER_ERR_MSG("create op[%s] failed", kNodeData.c_str());
      return DvppErrorCode::kFailed);

  ge::NodePtr data_node = else_graph->AddNode(data_op_desc_ptr);
  DVPP_CHECK_NOTNULL(data_node);

  ge::NodePtr decode_node = else_graph->AddNode(op_desc);
  DVPP_CHECK_NOTNULL(decode_node);
  decode_node->SetOwnerComputeGraph(else_graph);

  ret = InsertNodeAfter(data_node, decode_node);
  DVPP_CHECK_IF_THEN_DO(ret != DvppErrorCode::kSuccess,
      return DvppErrorCode::kFailed);

  // 打上 aicpu 标签
  auto decode_op_desc = decode_node->GetOpDesc();
  ret = SetAiCpuOpAttr(decode_op_desc);
  DVPP_CHECK_IF_THEN_DO(ret != DvppErrorCode::kSuccess,
      DVPP_REPORT_INNER_ERR_MSG("op[%s] set dvpp op attr failed",
                              decode_op_desc->GetType().c_str());
      return DvppErrorCode::kFailed);

  bool result = ge::AttrUtils::SetBool(decode_op_desc, ge::NEED_INFER, true);
  DVPP_CHECK_IF_THEN_DO(!result,
      DVPP_REPORT_INNER_ERR_MSG("set op[%s] need infer attr failed.",
                              decode_op_desc->GetType().c_str());
      return DvppErrorCode::kFailed);

  ge::NodePtr output_node = InsertGraphNodeAfter(else_graph, decode_node,
                                                 kNodeNetOutput);
  DVPP_CHECK_NOTNULL(output_node);

  ret = SetSubgraphDataIndex(data_node, 1, output_node, 0);
  DVPP_CHECK_IF_THEN_DO(ret != DvppErrorCode::kSuccess,
      DVPP_REPORT_INNER_ERR_MSG("set else subgraph data index failed.");
      return DvppErrorCode::kFailed);

  if (decode_node->GetType() == kDvppOpDecodeAndCropJpeg) {
    return InsertDataNodeInIfSubgraph(else_graph, decode_node);
  }

  return DvppErrorCode::kSuccess;
}

/*
 *                                                       Data
 *                                                        |
 *                            Data                     DecodeJpeg
 *                             |       then_subgraph      |
 *                       DecodeJpegPre   /              NetOutput
 *      Data                   |        /
 *       |                     If  -----
 *   DecodeJpeg  ===>          |        \                Data
 *       |                     |         \                |
 *    NetOutput             NetOutput   else_subgraph  DecodeJpeg
 *                                                        |
 *                                                     NetOutput
 */
DvppErrorCode DvppOptimizer::InsertIfGraph(
    ge::ComputeGraph& graph, ge::NodePtr& node) const {
  ge::ComputeGraphPtr graph_ptr = graph.shared_from_this();
  auto node_input_desc = node->GetOpDesc()->GetInputDesc("contents");
  auto node_output_desc = node->GetOpDesc()->GetOutputDesc("image");
  // insert DecodeJpegPre node
  ge::NodePtr pre_node = InsertGraphNodeBefore(graph_ptr, node,
                                               kDvppOpDecodeJpegPre);
  DVPP_CHECK_NOTNULL(pre_node);
  // insert If node
  ge::NodePtr if_node = InsertGraphNodeBefore(graph_ptr, node, kDvppOpIf);
  DVPP_CHECK_NOTNULL(if_node);
  // DecodeJpegPre node和If node之前增加一个数据连边
  auto input0_anchor = pre_node->GetInDataAnchor(0);
  DVPP_CHECK_NOTNULL(input0_anchor);
  auto input0_peer_anchor = input0_anchor->GetFirstPeerAnchor();
  auto input1_anchor = if_node->GetInDataAnchor(1);
  auto error = ge::GraphUtils::AddEdge(input0_peer_anchor, input1_anchor);
  DVPP_CHECK_IF_THEN_DO(error != ge::GRAPH_SUCCESS,
      DVPP_REPORT_INNER_ERR_MSG("add edge DecodeJpegPre input0 and If input1");
      return DvppErrorCode::kFailed);

  if (node->GetType() == kDvppOpDecodeAndCropJpeg) {
    // DecodeAndCropJpeg node input1和If node input2的数据连边，crop_window
    auto data1_anchor = node->GetInDataAnchor(1);
    DVPP_CHECK_NOTNULL(data1_anchor);
    auto data1_peer_anchor = data1_anchor->GetFirstPeerAnchor();
    auto data2_anchor = if_node->GetInDataAnchor(kNum2);
    error = ge::GraphUtils::AddEdge(data1_peer_anchor, data2_anchor);
    DVPP_CHECK_IF_THEN_DO(error != ge::GRAPH_SUCCESS,
        DVPP_REPORT_INNER_ERR_MSG("add edge Data node input0 and If input2");
        return DvppErrorCode::kFailed);
  }

  error = graph.RemoveNode(node);
  DVPP_CHECK_IF_THEN_DO(error != ge::GRAPH_SUCCESS,
      DVPP_REPORT_INNER_ERR_MSG("remove node failed, error:%d", error);
      return DvppErrorCode::kFailed);

  ge::OpDescPtr new_op_desc = ge::AttrUtils::CopyOpDesc(node->GetOpDesc());
  DVPP_CHECK_NOTNULL(new_op_desc);
  const std::string then_name = "dvpp_decode_if_then";
  ge::ComputeGraphPtr then_graph = nullptr;
  DVPP_TRY_CATCH(then_graph = std::make_shared<ge::ComputeGraph>(then_name),
      DVPP_REPORT_INNER_ERR_MSG("make shared then subgraph failed.");
      return DvppErrorCode::kFailed);

  auto ret = BuildIfThenGraph(then_graph, node);
  DVPP_CHECK_IF_THEN_DO(ret != DvppErrorCode::kSuccess,
      DVPP_REPORT_INNER_ERR_MSG("build then graph failed.");
      return DvppErrorCode::kFailed);
  then_graph->SetParentNode(if_node);
  then_graph->SetParentGraph(graph.shared_from_this());
  if_node->GetOpDesc()->AddSubgraphName(then_name);
  if_node->GetOpDesc()->SetSubgraphInstanceName(0, then_name);

  ge::ComputeGraphPtr root_graph =
      ge::GraphUtils::FindRootGraph(graph.shared_from_this());
  DVPP_CHECK_NOTNULL(root_graph);
  error = root_graph->AddSubgraph(then_graph);
  DVPP_CHECK_IF_THEN_DO(error != ge::GRAPH_SUCCESS,
      DVPP_REPORT_INNER_ERR_MSG("add then subgraph failed, error:%d", error);
      return DvppErrorCode::kFailed);

  const std::string else_name = "dvpp_decode_if_else";
  ge::ComputeGraphPtr else_graph = nullptr;
  DVPP_TRY_CATCH(else_graph = std::make_shared<ge::ComputeGraph>(else_name),
      DVPP_REPORT_INNER_ERR_MSG("make shared else subgraph failed.");
      return DvppErrorCode::kFailed);

  ret = BuildIfElseGraph(else_graph, new_op_desc);
  DVPP_CHECK_IF_THEN_DO(ret != DvppErrorCode::kSuccess,
      DVPP_REPORT_INNER_ERR_MSG("build then graph failed.");
      return DvppErrorCode::kFailed);
  else_graph->SetParentNode(if_node);
  else_graph->SetParentGraph(graph.shared_from_this());
  if_node->GetOpDesc()->AddSubgraphName(else_name);
  if_node->GetOpDesc()->SetSubgraphInstanceName(1, else_name);
  // 刷新if节点的input/output op desc
  if_node->GetOpDesc()->UpdateInputDesc(1, node_input_desc);
  if_node->GetOpDesc()->UpdateOutputDesc(0, node_output_desc);

  error = root_graph->AddSubgraph(else_graph);
  DVPP_CHECK_IF_THEN_DO(error != ge::GRAPH_SUCCESS,
      DVPP_REPORT_INNER_ERR_MSG("add else subgraph failed, error:%d", error);
      return DvppErrorCode::kFailed);

  std::string parent_graph_session_id{};
  ret = GetParentGraphSessionId(graph.shared_from_this(),
                                parent_graph_session_id);
  DVPP_CHECK_IF_THEN_DO(ret != DvppErrorCode::kSuccess,
      return DvppErrorCode::kFailed);
  ret = SetSubgraphParentSessionId(then_graph, else_graph,
                                   parent_graph_session_id);
  DVPP_CHECK_IF_THEN_DO(ret != DvppErrorCode::kSuccess,
      return DvppErrorCode::kFailed);

  return DvppErrorCode::kSuccess;
}

ge::NodePtr DvppOptimizer::GraphAddNode(ge::ComputeGraphPtr& graph,
    const std::string& op_type) const {
  // create op_desc
  ge::OpDescPtr op_desc_ptr = nullptr;
  auto ret = DvppOpsLib::Instance().CreateDvppOpDesc(op_type, op_desc_ptr);
  DVPP_CHECK_IF_THEN_DO(ret != DvppErrorCode::kSuccess,
      DVPP_REPORT_INNER_ERR_MSG("create op[%s] failed.", op_type.c_str());
      return nullptr);

  // add node in graph
  ge::NodePtr insert_node = graph->AddNode(op_desc_ptr);
  DVPP_CHECK_IF_THEN_DO(insert_node == nullptr,
      DVPP_REPORT_INNER_ERR_MSG("create insert node[%s] failed.",
                              op_type.c_str());
      return nullptr);

  return insert_node;
}

ge::NodePtr DvppOptimizer::GraphAddAICpuReduceMeanNode(
    ge::ComputeGraphPtr& graph) const {
  // create op_desc
  ge::OpDescPtr op_desc_ptr = nullptr;
  auto ret =  DvppOpsLib::Instance().CreateAICpuReduceMeanDesc(op_desc_ptr);
  DVPP_CHECK_IF_THEN_DO(ret != DvppErrorCode::kSuccess,
  DVPP_REPORT_INNER_ERR_MSG("create op[%s] failed.", kAICpuOpReduceMean.c_str());
      return nullptr);

  // add node in graph
  ge::NodePtr new_node = graph->AddNode(op_desc_ptr);
  DVPP_CHECK_IF_THEN_DO(new_node == nullptr,
  DVPP_REPORT_INNER_ERR_MSG("create insert node[%s] failed.",
      kAICpuOpReduceMean.c_str());
      return nullptr);

  return new_node;
}

DvppErrorCode DvppOptimizer::AddAdujustContrastWithMeanEdge(
    ge::NodePtr& contrast_with_mean_node, ge::NodePtr& contrast_node) const {
  // AdjustContrastWithMean节点替换原节点输出边
  auto contrast_node_out_anchor_0 = contrast_node->GetOutDataAnchor(kNum0);
  DVPP_CHECK_NOTNULL(contrast_node_out_anchor_0);
  auto contrast_with_mean_node_out_anchor_0 =
      contrast_with_mean_node->GetOutDataAnchor(kNum0);
  auto output_in_anchors =
      contrast_node_out_anchor_0->GetPeerInDataAnchors();
  DVPP_CHECK_IF_THEN_DO(output_in_anchors.size() == 0,
      DVPP_REPORT_INNER_ERR_MSG("output_in_anchors is empty.");
      return DvppErrorCode::kFailed);

  DVPP_CHECK_IF_THEN_DO(
      (ge::GraphUtils::RemoveEdge(contrast_node_out_anchor_0,
      output_in_anchors.at(kNum0)) != ge::GRAPH_SUCCESS) ||
      (ge::GraphUtils::AddEdge(contrast_with_mean_node_out_anchor_0,
      output_in_anchors.at(kNum0)) != ge::GRAPH_SUCCESS),
  DVPP_REPORT_INNER_ERR_MSG(
      "replace output edge between node[%s] and node[%s] failed.",
      kDvppOpAdjustContrastWithMean.c_str(), contrast_node->GetType().c_str());
      return DvppErrorCode::kFailed);
  // AdjustContrastWithMean节点替换原节点输入边0
  auto contrast_node_in_anchor_0 = contrast_node->GetInDataAnchor(kNum0);
  DVPP_CHECK_NOTNULL(contrast_node_in_anchor_0);
  auto contrast_with_mean_node_in_anchor_0 =
      contrast_with_mean_node->GetInDataAnchor(kNum0);
  auto contrast_with_mean_node_in_anchor_1 =
      contrast_with_mean_node->GetInDataAnchor(kNum1);
  auto data_out_anchor = contrast_node_in_anchor_0->GetPeerOutAnchor();
  DVPP_CHECK_IF_THEN_DO(
      (ge::GraphUtils::RemoveEdge(data_out_anchor,
      contrast_node_in_anchor_0) != ge::GRAPH_SUCCESS) ||
      (ge::GraphUtils::AddEdge(data_out_anchor,
      contrast_with_mean_node_in_anchor_0) != ge::GRAPH_SUCCESS) ||
      (ge::GraphUtils::AddEdge(data_out_anchor,
      contrast_with_mean_node_in_anchor_1) != ge::GRAPH_SUCCESS),
  DVPP_REPORT_INNER_ERR_MSG(
      "replace input image edge between node[%s] and node[%s] failed.",
      kDvppOpAdjustContrastWithMean.c_str(), contrast_node->GetType().c_str());
      return DvppErrorCode::kFailed);
  // AdjustContrastWithMean节点替换原节点输入边1
  auto contrast_node_in_anchor_1 = contrast_node->GetInDataAnchor(kNum1);
  DVPP_CHECK_NOTNULL(contrast_node_in_anchor_1);
  auto contrast_with_mean_node_in_anchor_2 =
      contrast_with_mean_node->GetInDataAnchor(kNum2);
  auto factor_out_anchor = contrast_node_in_anchor_1->GetPeerOutAnchor();
  DVPP_CHECK_IF_THEN_DO(
      (ge::GraphUtils::RemoveEdge(factor_out_anchor,
      contrast_node_in_anchor_1) != ge::GRAPH_SUCCESS) ||
      (ge::GraphUtils::AddEdge(factor_out_anchor,
      contrast_with_mean_node_in_anchor_2) != ge::GRAPH_SUCCESS),
      DVPP_REPORT_INNER_ERR_MSG(
          "replace input factor edge between node[%s] and node[%s] failed.",
          kDvppOpAdjustContrastWithMean.c_str(),
          contrast_node->GetType().c_str());
      return DvppErrorCode::kFailed);
  return DvppErrorCode::kSuccess;
}

ge::NodePtr DvppOptimizer::InsertAdujustContrastWithMean(
    ge::ComputeGraphPtr graph_ptr, ge::NodePtr& contrast_node) const {
  // 1.创建AdjustContrastWithMean节点并连边
  ge::NodePtr contrast_with_mean_node = GraphAddNode(graph_ptr,
      kDvppOpAdjustContrastWithMean);
  // 1.1 图中增加AdjustContrastWithMean节点
  DVPP_CHECK_IF_THEN_DO(contrast_with_mean_node == nullptr, return nullptr);
  // 1.2 连AdjustContrastWithMean的边
  DVPP_CHECK_IF_THEN_DO(AddAdujustContrastWithMeanEdge(
    contrast_with_mean_node, contrast_node) == DvppErrorCode::kFailed,
    return nullptr);
  // 1.3 获取并设置属性
  auto contrast_node_op_desc_ptr = contrast_node->GetOpDesc();
  auto contrast_with_mean_op_desc_ptr = contrast_with_mean_node->GetOpDesc();
  std::string attr_data_format_value = "";
  DVPP_CHECK_IF_THEN_DO(!ge::AttrUtils::GetStr(
      contrast_node_op_desc_ptr, "data_format", attr_data_format_value),
  DVPP_REPORT_INNER_ERR_MSG("get op[%s] data format failed.",
      contrast_node_op_desc_ptr->GetType().c_str());
      return nullptr);
  if (!attr_data_format_value.empty()) { // 原算子属性不为空时才需设置
    DVPP_CHECK_IF_THEN_DO(!ge::AttrUtils::SetStr(
        contrast_with_mean_op_desc_ptr, "data_format", attr_data_format_value),
    DVPP_REPORT_INNER_ERR_MSG("set op[%s] data format failed.",
        contrast_with_mean_op_desc_ptr->GetType().c_str());
        return nullptr);
  }
  // 5.设置新插入节点的输入输出描述
  DVPP_CHECK_IF_THEN_DO(
      (contrast_with_mean_op_desc_ptr->UpdateInputDesc(kNum0,
      contrast_node_op_desc_ptr->GetInputDesc(kNum0)) != ge::GRAPH_SUCCESS) ||
      (contrast_with_mean_op_desc_ptr->UpdateInputDesc(kNum1,
      contrast_with_mean_op_desc_ptr->GetInputDesc(kNum0)) !=
      ge::GRAPH_SUCCESS) ||
      (contrast_with_mean_op_desc_ptr->UpdateInputDesc(kNum2,
      contrast_node_op_desc_ptr->GetInputDesc(kNum1)) != ge::GRAPH_SUCCESS) ||
      (contrast_with_mean_op_desc_ptr->UpdateOutputDesc(kNum0,
      contrast_with_mean_op_desc_ptr->GetInputDesc(kNum0)) !=
      ge::GRAPH_SUCCESS),
      DVPP_REPORT_INNER_ERR_MSG("set node[%s] input or output description failed.",
      contrast_with_mean_node->GetType().c_str());
      return nullptr);
  return contrast_with_mean_node;
}

ge::NodePtr DvppOptimizer::InsertAICpuReduceMean(ge::ComputeGraphPtr graph_ptr,
    ge::NodePtr& contrast_node, ge::NodePtr& contrast_with_mean_node) const {
  // 2.插AICpuReduceMean算子到AdjustContrastWithMean算子的输入1之前
  ge::NodePtr reduce_mean_node = GraphAddAICpuReduceMeanNode(graph_ptr);
  // 2.1 插入算子，插入时输入输出位置取默认值，即AICpuReduceMean的0号输入和0号输出
  DVPP_CHECK_IF_THEN_DO(reduce_mean_node == nullptr,
      DVPP_REPORT_INNER_ERR_MSG("add node[%s] to graph failed.",
          kAICpuOpReduceMean.c_str());
      return nullptr);
  auto contrast_with_mean_node_in_anchor_1 = contrast_with_mean_node->
    GetInDataAnchor(kNum1);
  DVPP_CHECK_IF_THEN_DO(
      ge::GraphUtils::InsertNodeBefore(contrast_with_mean_node_in_anchor_1,
      reduce_mean_node) != ge::GRAPH_SUCCESS,
      DVPP_REPORT_INNER_ERR_MSG("insert node[%s] before node[%s] failed.",
      reduce_mean_node->GetType().c_str(),
      contrast_with_mean_node->GetType().c_str());
      return nullptr);
  // 2.2 打上 aicpu 标签
  auto reduce_mean_op_desc_ptr = reduce_mean_node->GetOpDesc();
  DVPP_CHECK_IF_THEN_DO(SetAiCpuOpAttr(reduce_mean_op_desc_ptr) !=
      DvppErrorCode::kSuccess,
      DVPP_REPORT_INNER_ERR_MSG("op[%s] set dvpp op attr failed",
      reduce_mean_node->GetType().c_str());
      return nullptr);
  DVPP_CHECK_IF_THEN_DO(!ge::AttrUtils::SetBool(
      reduce_mean_op_desc_ptr, ge::NEED_INFER, true),
      DVPP_REPORT_INNER_ERR_MSG("set op[%s] need infer attr failed.",
      reduce_mean_node->GetType().c_str());
      return nullptr);
  // 2.3 设置属性
  ge::AttrUtils::SetBool(reduce_mean_op_desc_ptr, "keep_dims", false);
  std::string attr_data_format_value = "";
  auto contrast_node_op_desc_ptr = contrast_node->GetOpDesc();
  DVPP_CHECK_IF_THEN_DO(!ge::AttrUtils::GetStr(
      contrast_node_op_desc_ptr, "data_format", attr_data_format_value),
  DVPP_REPORT_INNER_ERR_MSG("get op[%s] data format failed.",
      contrast_node_op_desc_ptr->GetType().c_str());
      return nullptr);
  if (!attr_data_format_value.empty()) { // 原算子属性不为空时才需设置
    DVPP_CHECK_IF_THEN_DO(!ge::AttrUtils::SetStr(
        reduce_mean_op_desc_ptr, "data_format", attr_data_format_value),
        DVPP_REPORT_INNER_ERR_MSG("set op[%s] data format attr failed.",
        reduce_mean_node->GetType().c_str());
        return nullptr);
  }
  // 2,4 设置新插入节点的输入输出描述
  DVPP_CHECK_IF_THEN_DO(
      (reduce_mean_op_desc_ptr->UpdateInputDesc(kNum0,
      contrast_node_op_desc_ptr->GetOutputDesc(kNum0)) != ge::GRAPH_SUCCESS) ||
      (reduce_mean_op_desc_ptr->UpdateOutputDesc(kNum0,
      reduce_mean_op_desc_ptr->GetInputDesc(kNum0)) != ge::GRAPH_SUCCESS),
      DVPP_REPORT_INNER_ERR_MSG("set node[%s] input or output desc failed.",
      reduce_mean_node->GetType().c_str());
      return nullptr);
  return reduce_mean_node;
}

ge::GeTensorPtr DvppOptimizer::GetConstTensorForReduceMean(
    ge::NodePtr& contrast_node) const{
  // 根据contrast节点输入的format或属性来推导reduce的维度
  int32_t n_index = 0, h_index = 0, w_index = 0;
  auto contrast_node_op_desc_ptr = contrast_node->GetOpDesc();
  auto tensor_data_format = contrast_node_op_desc_ptr->
      GetInputDesc(kNum0).GetFormat();
  std::string attr_data_format_value = "";
  DVPP_CHECK_IF_THEN_DO(!ge::AttrUtils::GetStr(
      contrast_node_op_desc_ptr, "data_format", attr_data_format_value),
  DVPP_REPORT_INNER_ERR_MSG("get op[%s] data format failed.",
      contrast_node_op_desc_ptr->GetType().c_str());
      return nullptr);
  if (attr_data_format_value == "HWC" ||
      tensor_data_format == ge::FORMAT_NHWC) { // 以attr中的data format的值优先
    h_index = 1; // H在第二维，下标是1
    w_index = 2; // W在第三维，下标是2
  } else { // FORMAT_NCHW 或 attr是"CHW"，其他情况进不来
    h_index = 2; // H在第三维，下标是2
    w_index = 3; // W在第四维，下标是3
  }
  std::vector<int32_t> reduce_mean_axes_tensor_value =
      {n_index, h_index, w_index};
  // 设置const节点的属性和值
  constexpr int64_t reduceMeanDim = 3;
  ge::GeTensorPtr reduce_mean_axes_tensor = MakeShared<ge::GeTensor>();
  auto reduce_mean_axes_shape =
      ge::GeShape(std::vector<int64_t>({reduceMeanDim}));
  reduce_mean_axes_tensor->MutableTensorDesc().SetDataType(ge::DT_INT32);
  reduce_mean_axes_tensor->MutableTensorDesc().SetFormat(ge::FORMAT_ND);
  reduce_mean_axes_tensor->MutableTensorDesc().SetShape(reduce_mean_axes_shape);
  reduce_mean_axes_tensor->MutableTensorDesc().SetOriginDataType(ge::DT_INT32);
  reduce_mean_axes_tensor->MutableTensorDesc().SetOriginFormat(ge::FORMAT_ND);
  reduce_mean_axes_tensor->
      MutableTensorDesc().SetOriginShape(reduce_mean_axes_shape);
  DVPP_CHECK_IF_THEN_DO((reduce_mean_axes_tensor->
      SetData(reinterpret_cast<uint8_t *>(reduce_mean_axes_tensor_value.data()),
      reduceMeanDim * sizeof(int32_t))) != ge::GRAPH_SUCCESS,
      DVPP_REPORT_INNER_ERR_MSG("set reduce mean axes data failed.");
      return nullptr);
  return reduce_mean_axes_tensor;
}

ge::NodePtr DvppOptimizer::InsertConstForReduceMean(ge::ComputeGraphPtr graph_ptr,
    ge::NodePtr& contrast_node, ge::NodePtr& reduce_mean_node) const{
  // 3.插AICpuReduceMean算子的const节点
  // 3.1 获取const tensor
  auto reduce_mean_axes_tensor = GetConstTensorForReduceMean(contrast_node);
  DVPP_CHECK_IF_THEN_DO(reduce_mean_axes_tensor == nullptr,
      return nullptr);
  // 3.2 增加const节点，即创建const节点、插入图、连边
  ge::OpDescPtr reduce_mean_axes_const_desc_ptr =
      ge::OpDescUtils::CreateConstOp(reduce_mean_axes_tensor);
      DVPP_CHECK_IF_THEN_DO(reduce_mean_axes_const_desc_ptr == nullptr,
      DVPP_REPORT_INNER_ERR_MSG("create reduce mean axes description failed.");
      return nullptr);

  auto reduce_mean_axes_const_node = graph_ptr->
      AddNode(reduce_mean_axes_const_desc_ptr);
      DVPP_CHECK_IF_THEN_DO(reduce_mean_axes_const_node == nullptr,
      DVPP_REPORT_INNER_ERR_MSG("create reduce mean axes node failed.");
      return nullptr);

  auto reduce_mean_axes_const_node_out_anchor_0 =
      reduce_mean_axes_const_node->GetOutDataAnchor(kNum0);
  auto reduce_mean_in_anchor_1 = reduce_mean_node->GetInDataAnchor(kNum1);
  DVPP_CHECK_IF_THEN_DO(
  (ge::GraphUtils::AddEdge(reduce_mean_axes_const_node_out_anchor_0,
      reduce_mean_in_anchor_1) != ge::GRAPH_SUCCESS),
      DVPP_REPORT_INNER_ERR_MSG("add edge between node[%s] and node[%s] failed.",
      reduce_mean_axes_const_node->GetType().c_str(),
      reduce_mean_node->GetType().c_str());
      return nullptr);
  // 3.3 设置reduce mean输入1属性
  auto reduce_mean_op_desc_ptr = reduce_mean_node->GetOpDesc();
  DVPP_CHECK_IF_THEN_DO(
      (reduce_mean_op_desc_ptr->UpdateInputDesc(1,
      reduce_mean_axes_const_desc_ptr->
      GetOutputDesc(kNum0)) != ge::GRAPH_SUCCESS),
      DVPP_REPORT_INNER_ERR_MSG("set node[%s] input 1 description failed.",
      reduce_mean_node->GetType().c_str()); return nullptr);
  return reduce_mean_axes_const_node;
}

ge::NodePtr DvppOptimizer::InsertRgbToGrayscale(ge::ComputeGraphPtr graph_ptr,
    ge::NodePtr& contrast_node, ge::NodePtr& reduce_mean_node) const{
  // 需要插入RgbToGrayscale节点，即插入节点、设置输入输出、设置属性
  ge::NodePtr rgb_to_gray_scale_node =
      InsertGraphNodeBefore(graph_ptr, reduce_mean_node, kDvppOpRgbToGrayscale);
  DVPP_CHECK_IF_THEN_DO(rgb_to_gray_scale_node == nullptr,
      DVPP_REPORT_INNER_ERR_MSG("insert RgbToGrayscale node failed");
      return nullptr);

  auto rgb_to_gray_scale_op_desc_ptr = rgb_to_gray_scale_node->GetOpDesc();
  DVPP_CHECK_IF_THEN_DO(rgb_to_gray_scale_op_desc_ptr == nullptr,
      DVPP_REPORT_INNER_ERR_MSG("get RgbToGrayscale op description failed");
      return nullptr);

  auto contrast_node_op_desc_ptr = contrast_node->GetOpDesc();
  DVPP_CHECK_IF_THEN_DO(contrast_node_op_desc_ptr == nullptr,
      DVPP_REPORT_INNER_ERR_MSG("get contrast_node desc failed");
      return nullptr);

  DVPP_CHECK_IF_THEN_DO(
      (rgb_to_gray_scale_op_desc_ptr->UpdateInputDesc(kNum0,
      contrast_node_op_desc_ptr->GetOutputDesc(kNum0)) != ge::GRAPH_SUCCESS) ||
      (rgb_to_gray_scale_op_desc_ptr->UpdateOutputDesc(kNum0,
      rgb_to_gray_scale_op_desc_ptr->GetInputDesc(kNum0)) != ge::GRAPH_SUCCESS),
      DVPP_REPORT_INNER_ERR_MSG(
      "set node[%s] input or output description failed.",
      rgb_to_gray_scale_node->GetType().c_str());
      return nullptr);

  constexpr int64_t outputChannels = 3;
  ge::AttrUtils::SetInt(rgb_to_gray_scale_op_desc_ptr,
      "output_channels", outputChannels);
  std::string attr_data_format_value = "";
  DVPP_CHECK_IF_THEN_DO(!ge::AttrUtils::GetStr(
      contrast_node_op_desc_ptr, "data_format", attr_data_format_value),
  DVPP_REPORT_INNER_ERR_MSG("get op[%s] data format failed.",
      contrast_node_op_desc_ptr->GetType().c_str());
      return nullptr);
  if (!attr_data_format_value.empty()) { // 原算子属性不为空时才需设置
    DVPP_CHECK_IF_THEN_DO(!ge::AttrUtils::SetStr(
        rgb_to_gray_scale_op_desc_ptr, "data_format", attr_data_format_value),
        DVPP_REPORT_INNER_ERR_MSG("set op[%s] data format attr failed.",
        kAICpuOpReduceMean.c_str());
        return nullptr);
  }
  return rgb_to_gray_scale_node;
}

/*
*                           Data
*                           /  \
* Data factor              /   R2G const factor
* \       /               /      \   /     /
* AdjustContrast ===>    /    ReduceMean  /
*     |                 /          \     /
* NetOutput           AdjustContrastWithMean
*                                |
*                            NetOutput
* 将adjust contrast算子替换为右边的图结构，用于实现adjust
* contrast功能。通过data计算出均值，输入到blend输入2可以
* 消除值依赖。R2G代表RgbToGrayscale，会根据attr判断是否插
* 入
*/
DvppErrorCode DvppOptimizer::InsertAdjustContrastGraph(
    ge::ComputeGraph& graph, ge::NodePtr& contrast_node) const {
  ge::ComputeGraphPtr graph_ptr = graph.shared_from_this();
  // 1.创建AdjustContrastWithMean节点并连边
  ge::NodePtr contrast_with_mean_node =
      InsertAdujustContrastWithMean(graph_ptr, contrast_node);
  DVPP_CHECK_IF_THEN_DO(contrast_with_mean_node == nullptr,
      return DvppErrorCode::kFailed);
  // 2.插AICpuReduceMean算子到AdjustContrastWithMean算子的输入1之前
  ge::NodePtr reduce_mean_node =
      InsertAICpuReduceMean(graph_ptr, contrast_node, contrast_with_mean_node);
  DVPP_CHECK_IF_THEN_DO(reduce_mean_node == nullptr,
      return DvppErrorCode::kFailed);
  // 3.插AICpuReduceMean算子的const节点
  ge::NodePtr reduce_mean_axes_const_node =
      InsertConstForReduceMean(graph_ptr, contrast_node, reduce_mean_node);
  DVPP_CHECK_IF_THEN_DO(reduce_mean_axes_const_node == nullptr,
      return DvppErrorCode::kFailed);
  // 4.根据mean_mode判断是否插入RgbToGrayscale节点
  std::string mean_mode_value = "";
  auto contrast_node_op_desc_ptr = contrast_node->GetOpDesc();
  DVPP_CHECK_IF_THEN_DO(!ge::AttrUtils::GetStr(contrast_node_op_desc_ptr,
      "mean_mode", mean_mode_value),
     DVPP_REPORT_INNER_ERR_MSG("set op[%s] mean mode attr failed.",
     kAICpuOpReduceMean.c_str());
     return DvppErrorCode::kFailed);
  if (mean_mode_value == "chn_y") {
    auto rgb_to_gray_scale_node = InsertRgbToGrayscale(graph_ptr,
        contrast_node, reduce_mean_node);
    DVPP_CHECK_IF_THEN_DO(rgb_to_gray_scale_node == nullptr,
      return DvppErrorCode::kFailed);
  }
  // mean_mode_value == "chn_wise",啥也不干
  DVPP_ENGINE_LOG_DEBUG("mean mode : %s", mean_mode_value.c_str());
  // 5.删除contrast_node
  DVPP_CHECK_IF_THEN_DO(graph.RemoveNode(contrast_node) != ge::GRAPH_SUCCESS,
      DVPP_REPORT_INNER_ERR_MSG("remove contrast node failed");
      return DvppErrorCode::kFailed);
  return DvppErrorCode::kSuccess;
  }

DvppErrorCode DvppOptimizer::InsertNode(
    ge::ComputeGraph& graph, ge::NodePtr& node) {
  std::string op_type = node->GetType();
  if (op_type == kDvppOpDecodeJpeg || op_type == kDvppOpDecodeAndCropJpeg) {
    // 在 DecodeJpeg 或 DecodeAndCropJpeg时插入If graph
    auto ret = InsertIfGraph(graph, node);
    DVPP_CHECK_IF_THEN_DO(ret != DvppErrorCode::kSuccess,
        DVPP_REPORT_INNER_ERR_MSG("insert if graph before node[%s] failed",
                                node->GetType().c_str());
        return ret);
  } else if (op_type == kDvppOpAdjustContrast) {
    // 在 AdjustContrast 时用 blend 算子实现
    auto ret = InsertAdjustContrastGraph(graph, node);
    DVPP_CHECK_IF_THEN_DO(ret != DvppErrorCode::kSuccess,
        DVPP_REPORT_INNER_ERR_MSG("insert if graph before node[%s] failed",
                                node->GetType().c_str());
        return ret);
  }

  return DvppErrorCode::kSuccess;
}

DvppErrorCode DvppOptimizer::OptimizeOriginalGraph(ge::ComputeGraph& graph) {
  for (auto& node : graph.GetDirectNode()) {
    DVPP_CHECK_NOTNULL(node);
    auto op_desc_ptr = node->GetOpDesc();
    DVPP_CHECK_NOTNULL(op_desc_ptr);
    bool enable_dvpp_engine = EnableDvppEngine(op_desc_ptr);
    DVPP_CHECK_IF_THEN_DO(!enable_dvpp_engine, continue);

    DVPP_ENGINE_LOG_DEBUG("dvpp engine is enable to op[%s].",
                          op_desc_ptr->GetType().c_str());

    // 特殊优化 必须在check support之前调用
    auto error = SpecialOptimize(graph, node);
    // 成功优化后则可以continue跳过后面步骤
    DVPP_CHECK_IF_THEN_DO(error == DvppErrorCode::kSuccess, continue);

    // check support
    std::string unsupported_reason;
    bool ret = DvppOpsChecker::CheckSupported(node, unsupported_reason);
    DVPP_CHECK_IF_THEN_DO(!ret, continue);

    DVPP_ENGINE_LOG_DEBUG("op[%s] CheckSupported success.",
                          op_desc_ptr->GetType().c_str());

    // 打上 dvpp 相关标签
    error = SetDvppOpAttr(op_desc_ptr);
    DVPP_CHECK_IF_THEN_DO(error != DvppErrorCode::kSuccess,
        DVPP_REPORT_INNER_ERR_MSG("op[%s] set dvpp op attr failed",
                                op_desc_ptr->GetType().c_str());
        return DvppErrorCode::kFailed);

    DVPP_ENGINE_LOG_DEBUG("op[%s] set dvpp op attr success.",
                          op_desc_ptr->GetType().c_str());

    // 根据情况 插入算子节点
    error = InsertNode(graph, node);
    DVPP_CHECK_IF_THEN_DO(error != DvppErrorCode::kSuccess,
        DVPP_REPORT_INNER_ERR_MSG("op[%s] insert node failed",
                                op_desc_ptr->GetType().c_str());
        return error);

    DVPP_ENGINE_LOG_DEBUG("op[%s] insert node success.",
                          op_desc_ptr->GetType().c_str());
  }

  return DvppErrorCode::kSuccess;
}

DvppErrorCode DvppOptimizer::OptimizeOriginalGraphJudgeInsert(
    ge::ComputeGraph& graph) {
  // 调用时机在checksupported之后 可以通过GetOpEngineName确认是否支持
  // 针对YUV对齐问题插入转换算子 当前没有业务需求
  (void)graph;
  return DvppErrorCode::kSuccess;
}

DvppErrorCode DvppOptimizer::OptimizeFusedGraph(ge::ComputeGraph& graph) const {
  // 调用时机在checksupported之后 可以通过GetOpEngineName确认是否支持
  // 算子融合 当前没有业务需求
  (void)graph;
  return DvppErrorCode::kSuccess;
}

DvppErrorCode DvppOptimizer::OptimizeWholeGraph(ge::ComputeGraph& graph) const {
  // 调用时机在checksupported之后 可以通过GetOpEngineName确认是否支持
  // 当前没有业务需求
  (void)graph;
  return DvppErrorCode::kSuccess;
}
} // namespace dvpp
