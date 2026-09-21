/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dflow/compiler/data_flow_graph/convert_batch_attr_to_udf_pass.h"

#include "common/compile_profiling/ge_trace_wrapper.h"
#include "common/plugin/ge_make_unique_util.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/ge_inner_error_codes.h"
#include "framework/common/types.h"
#include "graph/utils/graph_utils.h"
#include "dflow/flow_graph/data_flow_attr_define.h"
#include "dflow/compiler/data_flow_graph/data_flow_graph_utils.h"
#include "proto/dflow.pb.h"

namespace ge {
namespace {
// TimeBatch
constexpr const char *kTimeBatchOpType = "_BuiltIn_TimeBatch";
constexpr const char *kTimeBatchAttrWindow = "window";
constexpr const char *kTimeBatchAttrDim = "batch_dim";
constexpr const char *kTimeBatchAttrDropRemainder = "drop_remainder";
constexpr int64_t kDynamicWindow = -1;

// CountBatch
constexpr const char *kCountBatchOpType = "_BuiltIn_CountBatch";
constexpr const char *kCountBatchAttrBatchSize = "batch_size";
constexpr const char *kCountBatchAttrTimeout = "timeout";
constexpr const char *kCountBatchAttrPadding = "padding";
constexpr const char *kCountBatchAttrSlideStride = "slide_stride";
}  // namespace

bool ConvertBatchAttrToUdfPass::HasPeerOutAnchor(const Node &node, const int32_t index) const {
  auto in_data_anchor = node.GetInDataAnchor(index);
  if (in_data_anchor == nullptr) {
    return false;
  }
  if (in_data_anchor->GetPeerOutAnchor() != nullptr) {
    return true;
  }
  return false;
}

Status ConvertBatchAttrToUdfPass::GetTimeBatchAttrs(const GeTensorDesc &tensor_desc, int64_t &window, int64_t &dim,
                                                    bool &drop_remainder) const {
  GE_CHK_BOOL_RET_STATUS(AttrUtils::GetInt(tensor_desc, dflow::ATTR_NAME_TIME_BATCH_TIME_WINDOW, window), FAILED,
                         "Failed to get attr[%s].", dflow::ATTR_NAME_TIME_BATCH_TIME_WINDOW);
  (void)AttrUtils::GetInt(tensor_desc, dflow::ATTR_NAME_TIME_BATCH_BATCH_DIM, dim);
  (void)AttrUtils::GetBool(tensor_desc, dflow::ATTR_NAME_TIME_BATCH_DROP_REMAINDER, drop_remainder);
  return SUCCESS;
}

Status ConvertBatchAttrToUdfPass::GetCountBatchAttrs(const GeTensorDesc &tensor_desc,
                                                     CountBatchPara &batch_para) const {
  GE_CHK_BOOL_RET_STATUS(AttrUtils::GetInt(tensor_desc, dflow::ATTR_NAME_COUNT_BATCH_BATCH_SIZE, batch_para.batch_size),
                         FAILED, "Failed to get attr[%s].", dflow::ATTR_NAME_COUNT_BATCH_BATCH_SIZE);
  (void)AttrUtils::GetInt(tensor_desc, dflow::ATTR_NAME_COUNT_BATCH_TIMEOUT, batch_para.timeout);
  (void)AttrUtils::GetBool(tensor_desc, dflow::ATTR_NAME_COUNT_BATCH_PADDING, batch_para.padding);
  (void)AttrUtils::GetInt(tensor_desc, dflow::ATTR_NAME_COUNT_BATCH_SLIDE_STRIDE, batch_para.slide_stride);
  return SUCCESS;
}

Status ConvertBatchAttrToUdfPass::CheckTimeBatchAttrs(int64_t window) const {
  if ((window != kDynamicWindow) && (window <= 0)) {
    GELOGE(FAILED, "The attr[%s] value[%ld] is invalid, should be %ld or greater than 0.",
           dflow::ATTR_NAME_TIME_BATCH_TIME_WINDOW, window, kDynamicWindow);
    return FAILED;
  }
  return SUCCESS;
}

Status ConvertBatchAttrToUdfPass::CreateTimeBatchDesc(const Node &node, int32_t index,
                                                      OpDescPtr &time_batch_desc) const {
  // set default attr value for time batch
  int64_t window = -1;
  int64_t dim = -1;
  bool drop_remainder = false;

  auto op_desc = node.GetOpDesc();
  auto tensor_desc = op_desc->GetInputDesc(index);
  GE_CHK_STATUS_RET(GetTimeBatchAttrs(tensor_desc, window, dim, drop_remainder),
                    "Failed to get time batch attrs from node[%s] input[%d].", node.GetName().c_str(), index);

  GE_CHK_STATUS_RET(CheckTimeBatchAttrs(window), "The time batch attrs of node[%s] input[%d] is invalid.",
                    node.GetName().c_str(), index);

  // default 1 input and 1 output for time batch
  uint32_t in_out_num = 1U;
  std::string batch_op_name = node.GetName() + "_" + kTimeBatchOpType + "_" + std::to_string(index);
  GE_CHK_STATUS_RET(DataFlowGraphUtils::CreateFlowNodeOpDesc(batch_op_name, in_out_num, in_out_num, time_batch_desc),
                    "Failed to create time batch op desc for node[%s] input[%d].", node.GetName().c_str(), index);

  dataflow::ProcessPoint process_point;
  // create func
  dataflow::ProcessFunc process_func;
  process_func.set_name(kTimeBatchOpType);
  process_func.add_in_index(0);
  process_func.add_out_index(0);
  // create attrs
  std::map<std::string, proto::AttrDef> attrs;
  proto::AttrDef attr_window;
  attr_window.set_i(window);
  attrs[kTimeBatchAttrWindow] = attr_window;
  proto::AttrDef attr_dim;
  attr_dim.set_i(dim);
  attrs[kTimeBatchAttrDim] = attr_dim;
  proto::AttrDef attr_drop_remainder;
  attr_drop_remainder.set_b(drop_remainder);
  attrs[kTimeBatchAttrDropRemainder] = attr_drop_remainder;
  GE_CHK_STATUS_RET(
      DataFlowGraphUtils::CreateBuiltInFunctionProcessPoint(batch_op_name, {process_func}, attrs, process_point),
      "Failed to create process point for node[%s] input[%d].", node.GetName().c_str(), index);
  GE_CHK_STATUS_RET(DataFlowGraphUtils::BindProcessPointToFlowNode(process_point, time_batch_desc),
                    "Failed to bind process point[%s] to node[%s] input[%d].", batch_op_name.c_str(),
                    node.GetName().c_str(), index);

  GELOGD("Create TimeBatch for node[%s] input[%d] success, name = %s, window = %ld, dim = %ld, drop_remainder = %d.",
         node.GetName().c_str(), index, batch_op_name.c_str(), window, dim, drop_remainder);
  return SUCCESS;
}

Status ConvertBatchAttrToUdfPass::CreateCountBatchDesc(const Node &node, int32_t index,
                                                       OpDescPtr &count_batch_desc) const {
  auto op_desc = node.GetOpDesc();
  CountBatchPara batch_para = {0, 0, false, 0};
  auto tensor_desc = op_desc->GetInputDesc(index);
  GE_CHK_STATUS_RET(GetCountBatchAttrs(tensor_desc, batch_para),
                    "Failed to get count batch attrs from node[%s] input[%d].", node.GetName().c_str(), index);

  uint32_t in_out_num = 1U;
  std::string batch_op_name = node.GetName() + "_" + kCountBatchOpType + "_" + std::to_string(index);
  GE_CHK_STATUS_RET(DataFlowGraphUtils::CreateFlowNodeOpDesc(batch_op_name, in_out_num, in_out_num, count_batch_desc),
                    "Failed to create count batch op desc for node[%s] input[%d].", node.GetName().c_str(), index);

  dataflow::ProcessPoint process_point;
  // create func
  dataflow::ProcessFunc process_func;
  process_func.set_name(kCountBatchOpType);
  process_func.add_in_index(0);
  process_func.add_out_index(0);
  // create attrs
  std::map<std::string, proto::AttrDef> attrs;
  proto::AttrDef attr_batch_size;
  attr_batch_size.set_i(batch_para.batch_size);
  attrs[kCountBatchAttrBatchSize] = attr_batch_size;
  proto::AttrDef attr_timeout;
  attr_timeout.set_i(batch_para.timeout);
  attrs[kCountBatchAttrTimeout] = attr_timeout;
  proto::AttrDef attr_padding;
  attr_padding.set_b(batch_para.padding);
  attrs[kCountBatchAttrPadding] = attr_padding;
  proto::AttrDef attr_slide_stride;
  attr_slide_stride.set_i(batch_para.slide_stride);
  attrs[kCountBatchAttrSlideStride] = attr_slide_stride;
  GE_CHK_STATUS_RET(
      DataFlowGraphUtils::CreateBuiltInFunctionProcessPoint(batch_op_name, {process_func}, attrs, process_point),
      "Failed to create process point for node[%s] input[%d].", node.GetName().c_str(), index);
  GE_CHK_STATUS_RET(DataFlowGraphUtils::BindProcessPointToFlowNode(process_point, count_batch_desc),
                    "Failed to bind process point[%s] to node[%s] input[%d].", batch_op_name.c_str(),
                    node.GetName().c_str(), index);

  GELOGD(
      "Create CountBatch for node[%s] input[%d] success, name = %s, batch_size = %ld, timeout = %ld, padding = %d, "
      "slide_stride=%ld.",
      node.GetName().c_str(), index, batch_op_name.c_str(), batch_para.batch_size, batch_para.timeout,
      batch_para.padding, batch_para.slide_stride);
  return SUCCESS;
}

Status ConvertBatchAttrToUdfPass::UpdateAnchor(const Node &node, int32_t node_index, const Node &batch,
                                               int32_t batch_index) const {
  auto in_anchor = node.GetInDataAnchor(node_index);
  GE_CHECK_NOTNULL(in_anchor);
  auto peer_anchor = in_anchor->GetPeerOutAnchor();
  GE_CHECK_NOTNULL(peer_anchor);
  GE_CHECK_NOTNULL(peer_anchor->GetOwnerNode());
  GE_CHK_STATUS_RET(GraphUtils::RemoveEdge(peer_anchor, in_anchor),
                    "Failed to remove edge between node[%s] and node[%s] input[%d].",
                    peer_anchor->GetOwnerNode()->GetName().c_str(), node.GetName().c_str(), node_index);
  auto batch_out_anchor = batch.GetOutDataAnchor(batch_index);
  GE_CHECK_NOTNULL(batch_out_anchor);
  GE_CHK_STATUS_RET(GraphUtils::AddEdge(batch_out_anchor, in_anchor),
                    "Failed to add edge between node[%s] and node[%s] input[%d].", batch.GetName().c_str(),
                    node.GetName().c_str(), batch_index);
  auto batch_in_anchor = batch.GetInDataAnchor(batch_index);
  GE_CHECK_NOTNULL(batch_in_anchor);
  GE_CHK_STATUS_RET(GraphUtils::AddEdge(peer_anchor, batch_in_anchor),
                    "Failed to add edge between node[%s] and node[%s] input[%d].",
                    peer_anchor->GetOwnerNode()->GetName().c_str(), batch.GetName().c_str(), batch_index);
  return SUCCESS;
}

Status ConvertBatchAttrToUdfPass::InsertBatch(const Node &node, int32_t index, BatchType batch_type) const {
  OpDescPtr batch_desc = nullptr;
  if (batch_type == BatchType::kTimeBatch) {
    GE_CHK_STATUS_RET(CreateTimeBatchDesc(node, index, batch_desc), "Failed to create time batch for node[%s].",
                      node.GetName().c_str());
  } else if (batch_type == BatchType::kCountBatch) {
    GE_CHK_STATUS_RET(CreateCountBatchDesc(node, index, batch_desc), "Failed to create count batch for node[%s].",
                      node.GetName().c_str());
  } else {
    GELOGE(FAILED, "Failed to create batch for node[%s], because batch_type is invalid", node.GetName().c_str());
    return FAILED;
  }
  constexpr const char *kExtAttrDeployAffinityNode = "_data_flow_deploy_affinity_node";
  GE_CHK_BOOL_RET_STATUS(batch_desc->SetExtAttr(kExtAttrDeployAffinityNode, node.GetName()), FAILED,
                         "set ext attr[%s] to batch[%s] failed", kExtAttrDeployAffinityNode,
                         batch_desc->GetName().c_str());
  auto compute_graph = node.GetOwnerComputeGraph();
  GE_CHECK_NOTNULL(compute_graph);
  auto batch_node = compute_graph->AddNode(batch_desc);
  GE_CHECK_NOTNULL(batch_node);
  GE_CHK_STATUS_RET(UpdateAnchor(node, index, *batch_node, 0), "Failed update anchor[%d] for node[%s].", index,
                    node.GetName().c_str());

  return SUCCESS;
}

Status ConvertBatchAttrToUdfPass::Run(ge::ComputeGraphPtr graph) {
  GE_CHECK_NOTNULL(graph);
  bool exception_catch = false;
  (void)AttrUtils::GetBool(graph, dflow::ATTR_NAME_DATA_FLOW_ENABLE_EXCEPTION_CATCH, exception_catch);
  GE_TRACE_START(ConvertBatchAttrToUdfPass);
  BatchType batch_type = BatchType::kInvalidBatch;
  bool insert_batch_node = false;
  for (const auto &node : graph->GetDirectNode()) {
    GE_CHECK_NOTNULL(node);
    if (node->GetType() != FLOWNODE) {
      continue;
    }
    auto op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    const auto &input_descs = op_desc->GetAllInputsDescPtr();
    for (size_t i = 0; i < input_descs.size(); ++i) {
      bool input_has_time_batch_attr = AttrUtils::HasAttr(input_descs.at(i), dflow::ATTR_NAME_TIME_BATCH_TIME_WINDOW);
      bool input_has_count_batch_attr = AttrUtils::HasAttr(input_descs.at(i), dflow::ATTR_NAME_COUNT_BATCH_BATCH_SIZE);
      if (input_has_time_batch_attr && input_has_count_batch_attr) {
        GELOGE(FAILED, "TimeBatch and CountBatch can't be set on the same node[%s] input[%zu] at the same time.",
               node->GetName().c_str(), i);
        return FAILED;
      }
      if (exception_catch && (input_has_time_batch_attr || input_has_count_batch_attr)) {
        GELOGE(FAILED, "TimeBatch or CountBatch can't be set in node[%s] input[%zu] while exception catch is enable.",
               node->GetName().c_str(), i);
        return FAILED;
      }
      batch_type = input_has_time_batch_attr
                       ? BatchType::kTimeBatch
                       : (input_has_count_batch_attr ? BatchType::kCountBatch : BatchType::kInvalidBatch);
      if (input_has_time_batch_attr || input_has_count_batch_attr) {
        GE_CHK_BOOL_RET_STATUS(HasPeerOutAnchor(*node, static_cast<int32_t>(i)), FAILED,
                               "Could not batch for node[%s], input[%zu], batch_type[%u].", node->GetName().c_str(), i,
                               static_cast<uint32_t>(batch_type));
        GE_CHK_STATUS_RET(InsertBatch(*node, static_cast<int32_t>(i), batch_type),
                          "Failed to batch for node[%s], input[%zu], batch_type[%u].", node->GetName().c_str(), i,
                          static_cast<uint32_t>(batch_type));
        insert_batch_node = true;
      }
    }
  }
  if (insert_batch_node) {
    GE_CHK_STATUS_RET(DataFlowGraphUtils::EnsureNMappingAttr(graph),
                      "insert batch node, but failed to set n-mapping attr for graph[%s]", graph->GetName().c_str());
  }
  GE_DUMP(graph, "AfterConvertBatchAttrToUdfPass");
  std::string trace_log = "Run batch pass for graph[" + graph->GetName() + "] during building";
  GE_COMPILE_TRACE_TIMESTAMP_END(ConvertBatchAttrToUdfPass, trace_log.c_str());
  return SUCCESS;
}

REG_PASS_OPTION("ConvertBatchAttrToUdfPass").LEVELS(OoLevel::kO0);
}  // namespace ge
