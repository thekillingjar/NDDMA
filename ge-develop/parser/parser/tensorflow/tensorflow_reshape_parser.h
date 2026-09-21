/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PARSER_TENSORFLOW_TENSORFLOW_RESHAPE_PARSER_H_
#define PARSER_TENSORFLOW_TENSORFLOW_RESHAPE_PARSER_H_

#include "parser/tensorflow/tensorflow_op_parser.h"

namespace ge {
class PARSER_FUNC_VISIBILITY TensorFlowReshapeParser : public TensorFlowOpParser {
 private:
  static Status ParseDesc(const domi::tensorflow::AttrValue &attr_value, ge::GeTensorDesc &ge_desc);

 public:
  /**
  * @ingroup domi_omg
  * @brief parse weight information
  * @param [in] v_input_const weight data to be parsed
  * @param [out] op_dest weight data after parsing
  * @return SUCCESS parse successfully
  * @return FAILED parse failed
  * @author
  */
  Status ParseParams(const Message *op_src, ge::OpDescPtr &op_dest) override;
};
}  // namespace ge

#endif  // PARSER_TENSORFLOW_TENSORFLOW_RESHAPE_PARSER_H_
