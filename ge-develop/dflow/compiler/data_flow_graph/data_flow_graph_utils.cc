/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dflow/compiler/data_flow_graph/data_flow_graph_utils.h"
#include "common/util/mem_utils.h"
#include "framework/common/types.h"
#include "dflow/flow_graph/data_flow_attr_define.h"
#include "graph_metadef/common/ge_common/util.h"

namespace ge {
Status DataFlowGraphUtils::CreateFlowNodeOpDesc(const std::string &op_name, uint32_t input_num, uint32_t output_num,
                                                OpDescPtr &op_desc) {
  op_desc = MakeShared<OpDesc>(op_name, FLOWNODE);
  GE_CHECK_NOTNULL(op_desc);
  GE_CHK_STATUS_RET(op_desc->AddDynamicInputDesc(dflow::ATTR_NAME_DATA_FLOW_INPUT, input_num),
                    "Failed to add inputs for op[%s], inputs num[%u].", op_name.c_str(), input_num);
  GE_CHK_STATUS_RET(op_desc->AddDynamicOutputDesc(dflow::ATTR_NAME_DATA_FLOW_OUTPUT, output_num),
                    "Failed to add outputs for op[%s], outputs num[%u].", op_name.c_str(), output_num);
  return SUCCESS;
}

Status DataFlowGraphUtils::CreateFlowFuncOpDesc(const std::string &op_name, uint32_t input_num, uint32_t output_num,
                                                OpDescPtr &op_desc) {
  op_desc = MakeShared<OpDesc>(op_name, FLOWFUNC);
  GE_CHECK_NOTNULL(op_desc);
  GE_CHK_BOOL_RET_STATUS(AttrUtils::SetInt(op_desc, "__dynamic_input_x_cnt", input_num), FAILED,
                         "Failed to set attr[__dynamic_input_x_cnt] for op[%s], value[%u].", op_name.c_str(),
                         input_num);
  GE_CHK_STATUS_RET(op_desc->AddDynamicInputDesc("x", input_num),
                    "Failed to set FlowFunc inputs desc for op[%s], inputs num [%u].", op_name.c_str(), input_num);
  GE_CHK_BOOL_RET_STATUS(AttrUtils::SetInt(op_desc, "__dynamic_output_x_cnt", output_num), FAILED,
                         "Failed to set attr[__dynamic_output_x_cnt] for op[%s], value[%u].", op_name.c_str(),
                         output_num);
  GE_CHK_STATUS_RET(op_desc->AddDynamicOutputDesc("y", output_num),
                    "Failed to set FlowFunc outputs for op[%s], outputs num [%u].", op_name.c_str(), output_num);
  return SUCCESS;
}

Status DataFlowGraphUtils::CreateBuiltInFunctionProcessPoint(const std::string &process_point_name,
                                                             const std::vector<dataflow::ProcessFunc> &udf_funcs,
                                                             const std::map<std::string, proto::AttrDef> &attrs,
                                                             dataflow::ProcessPoint &process_point) {
  process_point.set_name(process_point_name);
  process_point.set_type(dataflow::ProcessPoint_ProcessPointType_FUNCTION);
  process_point.set_is_built_in(true);
  for (const auto &udf_func : udf_funcs) {
    auto *func = process_point.add_funcs();
    GE_CHECK_NOTNULL(func);
    *func = udf_func;
  }
  auto *process_point_attrs = process_point.mutable_attrs();
  GE_CHECK_NOTNULL(process_point_attrs);
  for (const auto &attr : attrs) {
    (*process_point_attrs)[attr.first] = attr.second;
  }
  return SUCCESS;
}

Status DataFlowGraphUtils::BindProcessPointToFlowNode(dataflow::ProcessPoint &process_point, OpDescPtr &op_desc) {
  GE_CHECK_NOTNULL(op_desc);
  std::string name = op_desc->GetName();
  const auto input_size = op_desc->GetInputsSize();
  for (size_t i = 0U; i < input_size; ++i) {
    auto in_edge = process_point.add_in_edges();
    GE_CHECK_NOTNULL(in_edge);
    in_edge->set_node_name(name);
    in_edge->set_index(static_cast<int64_t>(i));
  }
  const auto output_size = op_desc->GetOutputsSize();
  for (size_t i = 0U; i < output_size; ++i) {
    auto out_edge = process_point.add_out_edges();
    GE_CHECK_NOTNULL(out_edge);
    out_edge->set_node_name(name);
    out_edge->set_index(static_cast<int64_t>(i));
  }
  std::string process_point_msg;
  GE_CHK_BOOL_RET_STATUS(process_point.SerializeToString(&process_point_msg), FAILED,
                         "Failed to serialize process point[%s] to string.", process_point.name().c_str());
  std::vector<std::string> attr_pps = {process_point_msg};
  GE_CHK_BOOL_RET_STATUS(AttrUtils::SetListStr(op_desc, dflow::ATTR_NAME_DATA_FLOW_PROCESS_POINTS, attr_pps), FAILED,
                         "Failed to set attr[%s] for node[%s].", dflow::ATTR_NAME_DATA_FLOW_PROCESS_POINTS,
                         name.c_str());
  return SUCCESS;
}

Status DataFlowGraphUtils::EnsureNMappingAttr(const ComputeGraphPtr &graph) {
  bool contains_n_mapping_node = false;
  (void)AttrUtils::GetBool(graph, dflow::ATTR_NAME_DATA_FLOW_CONTAINS_N_MAPPING_NODE, contains_n_mapping_node);
  if (!contains_n_mapping_node) {
    GE_CHK_BOOL_RET_STATUS(AttrUtils::SetBool(graph, dflow::ATTR_NAME_DATA_FLOW_CONTAINS_N_MAPPING_NODE, true), FAILED,
                           "Failed to set attr[%s] to graph[%s]", dflow::ATTR_NAME_DATA_FLOW_CONTAINS_N_MAPPING_NODE,
                           graph->GetName().c_str());
    GELOGI("set attr[%s] to graph[%s] success.", dflow::ATTR_NAME_DATA_FLOW_CONTAINS_N_MAPPING_NODE,
           graph->GetName().c_str());
  }
  return SUCCESS;
}
}  // namespace ge
