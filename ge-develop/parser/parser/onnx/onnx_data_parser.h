/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_PARSER_ONNX_ONNX_DATA_PARSER_H_
#define GE_PARSER_ONNX_ONNX_DATA_PARSER_H_

#include <string>
#include <vector>
#include "parser/common/data_op_parser.h"
#include "parser/onnx/onnx_op_parser.h"

namespace ge {
class PARSER_FUNC_VISIBILITY OnnxDataParser : public OnnxOpParser {
 public:
  Status ParseParams(const Message *op_src, ge::Operator &op_def) override;

 private:
  Status ParseInputFromModel(const Message *op_src, ge::Operator &op_def);

  Status ParseInputFromUser(const ge::Operator &op_def);

  bool IsSubgraphDataOp() const {
    return is_subgraph_data_op_;
  }

  int64_t ParseInputTensor(const ge::onnx::AttributeProto &attribute);

  std::vector<int64_t> model_input_dims_v_;

  std::vector<int64_t> user_input_dims_v_;

  bool is_subgraph_data_op_ = false;
};
}  // namespace ge

#endif  // GE_PARSER_ONNX_ONNX_DATA_PARSER_H_
