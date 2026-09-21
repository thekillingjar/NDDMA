/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_PARSER_TENSORFLOW_TENSORFLOW_DATA_PARSER_H_
#define GE_PARSER_TENSORFLOW_TENSORFLOW_DATA_PARSER_H_

#include <string>
#include <vector>
#include "parser/common/data_op_parser.h"
#include "parser/tensorflow/tensorflow_op_parser.h"

namespace ge {
class PARSER_FUNC_VISIBILITY TensorFlowDataParser : public TensorFlowOpParser, public DataOpParser {
 public:
  /**
  * @ingroup domi_omg
  * @brief parse weight
  * @param [in] v_input_const weight data to be parsed
  * @param [out] op_dest weight data after parsing
  * @return SUCCESS parse successfully
  * @return FAILED parse failed
  * @author
  */
  Status ParseParams(const Message *op_src, ge::OpDescPtr &op_def) override;

 private:
  /**
  * @ingroup domi_omg
  * @brief Parsing input from model
  * @param [in] op_src model to be parsed
  * @param [out] op_def input information after parsing
  * @return SUCCESS parse successfully
  * @return FAILED parse failed
  * @author
  */
  Status ParseInputFromModel(const Message *op_src, const ge::OpDescPtr &op_def);

  /**
  * @ingroup domi_omg
  * @brief parse input set by users
  * @param [in] op_src model to be parsed
  * @param [out] op_def input information after parsing
  * @return SUCCESS parse successfully
  * @return FAILED parse failed
  * @author
  */
  Status ParseInputFromUser(const Message *op_src, const ge::OpDescPtr &op_def);

  /**
  * @ingroup domi_omg
  * @brief Check whether the input shape entered by the user matches the input shape defined by the model
  * @return SUCCESS match
  * @return FAILED not match
  * @author
  */
  Status CheckInputShape(const std::string &name);

  std::vector<int64_t> model_input_dims_v;

  std::vector<int64_t> user_input_dims_v;
};
}  // namespace ge

#endif  // GE_PARSER_TENSORFLOW_TENSORFLOW_DATA_PARSER_H_
