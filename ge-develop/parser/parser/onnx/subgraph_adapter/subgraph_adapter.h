/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_PARSER_ONNX_SUBGRAPH_ADAPTER_SUBGRAPH_ADAPTER_H_
#define GE_PARSER_ONNX_SUBGRAPH_ADAPTER_SUBGRAPH_ADAPTER_H_

#if defined(_MSC_VER)
#ifdef FUNC_VISIBILITY
#define PARSER_FUNC_VISIBILITY _declspec(dllexport)
#else
#define PARSER_FUNC_VISIBILITY
#endif
#else
#ifdef FUNC_VISIBILITY
#define PARSER_FUNC_VISIBILITY __attribute__((visibility("default")))
#else
#define PARSER_FUNC_VISIBILITY
#endif
#endif

#include <map>
#include <vector>
#include "proto/onnx/ge_onnx.pb.h"
#include "register/register_error_codes.h"
#include "framework/omg/parser/parser_types.h"
namespace ge {
class PARSER_FUNC_VISIBILITY SubgraphAdapter {
 public:
  SubgraphAdapter() = default;
  virtual ~SubgraphAdapter() = default;
  /// @brief parse params
  /// @param [in/out] parent_op               parent op
  /// @param [in/out] onnx_graph_tasks        onnx graph task
  /// @param [in/out] name_to_onnx_graph      map name to onnx graph
  /// @return SUCCESS                         parse success
  /// @return FAILED                          Parse failed
  virtual domi::Status AdaptAndFindAllSubgraphs(ge::onnx::NodeProto *parent_op,
                                          std::vector<ge::onnx::GraphProto *> &onnx_graphs,
                                          std::map<std::string, ge::onnx::GraphProto *> &name_to_onnx_graph,
                                          const std::string &parent_graph_name = "") {
    (void)parent_op;
    (void)onnx_graphs;
    (void)name_to_onnx_graph;
    (void)parent_graph_name;
    return domi::SUCCESS;
  }
};
}  // namespace ge

#endif  // GE_PARSER_ONNX_SUBGRAPH_ADAPTER_SUBGRAPH_ADAPTER_H_
