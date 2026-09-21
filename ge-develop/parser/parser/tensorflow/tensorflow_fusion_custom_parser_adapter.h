/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_PARSER_TENSORFLOW_TENSORFLOW_FUSION_CUSTOM_PARSER_ADAPTER_H_
#define GE_PARSER_TENSORFLOW_TENSORFLOW_FUSION_CUSTOM_PARSER_ADAPTER_H_

#include "parser/tensorflow/tensorflow_fusion_op_parser.h"

namespace ge {
class PARSER_FUNC_VISIBILITY TensorFlowFusionCustomParserAdapter : public TensorFlowFusionOpParser {
 public:
  /**
  * @ingroup domi_parser
  * @brief Parsing model file information
  * @param [in] v_input_const model data to be parsed
  * @param [out] node model data after parsing
  * @return SUCCESS parse successfully
  * @return FAILED parse failed
  * @author
  */
  Status ParseParams(const vector<const NodeDef *> &v_input_const, ge::NodePtr &node) const override;

  using TensorFlowFusionOpParser::ParseParams;
  /**
  * @ingroup domi_parser
  * @brief Parsing model file information
  * @param [in] v_input_const ge operators which save model data to be parsed
  * @param [out] node model data after parsing
  * @return SUCCESS parse successfully
  * @return FAILED parse failed
  * @author
  */
  Status ParseParams(const std::vector<ge::Operator> &v_input_const, ge::NodePtr &node) const;
};
}  // namespace ge

#endif  // GE_PARSER_TENSORFLOW_TENSORFLOW_FUSION_CUSTOM_PARSER_ADAPTER_H_
