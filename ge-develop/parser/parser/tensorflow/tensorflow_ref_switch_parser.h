/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DOMI_OMG_PARSER_OP_PARSER_TENSORFLOW_REF_SWITCH_H_
#define DOMI_OMG_PARSER_OP_PARSER_TENSORFLOW_REF_SWITCH_H_

#include "common/op_def/ref_switch_operator.h"
#include "parser/tensorflow/tensorflow_op_parser.h"

namespace ge {
class PARSER_FUNC_VISIBILITY TensorFlowRefSwitchParser : public TensorFlowOpParser {
  // AUTO GEN PLEASE DO NOT MODIFY IT
 public:
  /**
   * @ingroup domi_omg
   * @brief 解析模型文件信息
   * @param [in] v_input_const 待解析的模型数据
   * @param [out] node 解析后的模型数据
   * @return SUCCESS 解析成功
   * @return FAILED 解析失败
   */
  Status ParseParams(const Message *op_src, ge::OpDescPtr &op_dest) override;

 protected:
  /**
   * @ingroup domi_omg
   * @brief 解析模型文件信息
   * @param [in] v_input_const 待解析的模型数据
   * @param [out] node 解析后的模型数据
   * @return SUCCESS 解析成功
   * @return FAILED 解析失败
   */
  static Status ParseT(const domi::tensorflow::NodeDef *node, RefSwitchOperator *op);

  // AUTO GEN PLEASE DO NOT MODIFY IT
};
}  // namespace ge

#endif  // DOMI_OMG_PARSER_OP_PARSER_TENSORFLOW_REF_SWITCH_H_
