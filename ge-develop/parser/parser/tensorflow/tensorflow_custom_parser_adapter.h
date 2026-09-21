/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_PARSER_TENSORFLOW_TENSORFLOW_CUSTOM_PARSER_ADAPTER_H_
#define GE_PARSER_TENSORFLOW_TENSORFLOW_CUSTOM_PARSER_ADAPTER_H_

#include "parser/tensorflow/tensorflow_op_parser.h"

namespace ge {
class PARSER_FUNC_VISIBILITY TensorFlowCustomParserAdapter : public TensorFlowOpParser {
 public:
  /**
  * @ingroup domi_omg
  * @brief Parsing model file information
  * @param [in] op_src model data to be parsed
  * @param [out] op_dest model data after parsing
  * @return SUCCESS parse successfully
  * @return FAILED parse failed
  * @author
  */
  Status ParseParams(const Message *op_src, ge::OpDescPtr &op_dest) override;

  using TensorFlowOpParser::ParseParams;
   /**
   * @ingroup domi_omg
   * @brief parse params of the operation
   * @param [in] op_src params to be parsed
   * @param [out] op_dest params after parsing
   * @return SUCCESS parse successfully
   * @return FAILED parse failed
   * @author
   */
   Status ParseParams(const Operator &op_src, ge::OpDescPtr &op_dest) const;
};
}  // namespace ge

#endif  // GE_PARSER_TENSORFLOW_TENSORFLOW_CUSTOM_PARSER_ADAPTER_H_
