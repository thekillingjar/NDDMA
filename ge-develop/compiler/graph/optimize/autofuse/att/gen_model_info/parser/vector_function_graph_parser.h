/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_VECTOR_FUNCTION_GRAPH_PARSER_H
#define AUTOFUSE_VECTOR_FUNCTION_GRAPH_PARSER_H

#include "ascendc_ir_core/ascendc_ir.h"
#include "gen_model_info/parser/tuning_space.h"

namespace att {
class VectorFunctionGraphParser {
 public:
  VectorFunctionGraphParser(const ge::AscNodePtr &asc_node, const ge::AscGraph &graph)
      : asc_node_(asc_node), graph_(graph) {}
  ~VectorFunctionGraphParser() = default;
  ge::Status Parse();
  [[nodiscard]] const std::vector<NodeInfo> &GetNodesInfos() const { return nodes_infos_; }

 private:
  ge::Status ParseNodeInfos(NodeInfo &node_info);
  ge::Status ParseInputTensors(NodeInfo &node_info);
  ge::Status ParseOutputTensors(NodeInfo &node_info);
  ge::Status GetVectorizedAxes(const TensorPtr &tensor, const ge::AscTensorAttr &tensor_attr) const;
  ge::Status ParseTensorInfo(const ge::AscTensorAttr &attr, const TensorPtr &tensor, size_t index);

  std::vector<NodeInfo> nodes_infos_;
  const ge::AscNodePtr &asc_node_;
  const ge::AscGraph &graph_;
};
}  // namespace att

#endif  // AUTOFUSE_VECTOR_FUNCTION_GRAPH_PARSER_H
