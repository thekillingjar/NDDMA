/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PARSER_CAFFE_CAFFE_CUSTOM_PARSER_ADAPTER_H_
#define PARSER_CAFFE_CAFFE_CUSTOM_PARSER_ADAPTER_H_

#include "parser/caffe/caffe_op_parser.h"

namespace ge {
class PARSER_FUNC_VISIBILITY CaffeCustomParserAdapter : public CaffeOpParser {
 public:
  /**
   * @ingroup domi_omg
   * @brief parse params of the operation
   * @param [in] op_src params to be parsed
   * @param [out] op_dest params after parsing
   * @return SUCCESS parse successfully
   * @return FAILED parse failed
   * @author
   */
  Status ParseParams(const Message *op_src, ge::OpDescPtr &op_dest) override;

  /**
   * @ingroup domi_omg
   * @brief parse params of the operation
   * @param [in] op_src params to be parsed
   * @param [out] op_dest params after parsing
   * @return SUCCESS parse successfully
   * @return FAILED parse failed
   * @author
   */
   static Status ParseParams(const Operator &op_src, const ge::OpDescPtr &op_dest);

  /**
   * @ingroup domi_omg
   * @brief parse weight of the operation
   * @param [in] op_src params to be parsed
   * @param [out] node params after parsing
   * @return SUCCESS parse successfullyparse failed
   * @return FAILED
   * @author
   */
  Status ParseWeights(const Message *op_src, ge::NodePtr &node) override;

  /**
   * @ingroup domi_omg
   * @brief parse weight of the operation
   * @param [in] const_node const node to add link edge
   * @param [in] index index of current node to add link
   * @param [in] update_in_turn flag of update in turn
   * @param [out] node params after parsing
   * @return SUCCESS parse successfullyparse failed
   * @return FAILED
   * @author
   */
Status AddEdgeFromConstNode(const NodePtr &const_node, const int32_t index,
                          const bool update_in_turn, ge::NodePtr &node) const;
};
}  // namespace ge

#endif  // PARSER_CAFFE_CAFFE_CUSTOM_PARSER_ADAPTER_H_
