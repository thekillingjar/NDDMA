/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_COMPILER_PNE_DATA_FLOW_GRAPH_DATA_FLOW_GRAPH_UTILS_H
#define AIR_COMPILER_PNE_DATA_FLOW_GRAPH_DATA_FLOW_GRAPH_UTILS_H

#include <map>
#include <string>
#include <vector>
#include "dflow/compiler/data_flow_graph/data_flow_graph.h"
#include "proto/dflow.pb.h"

namespace ge {
class DataFlowGraphUtils {
 public:
  static Status CreateFlowNodeOpDesc(const std::string &op_name, uint32_t input_num, uint32_t output_num,
                                     OpDescPtr &op_desc);

  static Status CreateFlowFuncOpDesc(const std::string &op_name, uint32_t input_num, uint32_t output_num,
                                     OpDescPtr &op_desc);

  static Status CreateBuiltInFunctionProcessPoint(const std::string &process_point_name,
                                                  const std::vector<dataflow::ProcessFunc> &udf_funcs,
                                                  const std::map<std::string, proto::AttrDef> &attrs,
                                                  dataflow::ProcessPoint &process_point);

  static Status BindProcessPointToFlowNode(dataflow::ProcessPoint &process_point, OpDescPtr &op_desc);

  static Status EnsureNMappingAttr(const ComputeGraphPtr &graph);
};
}  // namespace ge

#endif
