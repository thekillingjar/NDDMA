/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PARSER_CAFFE_CAFFE_DATA_PARSER_H_
#define PARSER_CAFFE_CAFFE_DATA_PARSER_H_

#include <string>
#include <vector>
#include "parser/caffe/caffe_op_parser.h"
#include "parser/common/data_op_parser.h"

namespace ge {
class PARSER_FUNC_VISIBILITY CaffeDataParser : public CaffeOpParser, public DataOpParser {
 public:
  /**
   * @ingroup domi_omg
   * @brief parse params of the operation
   * @param [in] op_src params to be parsed
   * @param [out] graph params after parsing
   * @return SUCCESS parse successfully
   * @return FAILED parse failed
   */
  Status ParseParams(const Message *op_src, ge::OpDescPtr &op) override;

 private:
  /**
   * @ingroup domi_omg
   * @brief Get the output dimension according to the input dimension
   * @param [in] name the name of the input layer
   * @param [in] input_dims the dimension of the input layer
   * @param [out] op_def op after parsing
   * @return SUCCESS parse successfully
   * @return FAILED parse failed
   */
  Status GetOutputDesc(const std::string &name, const std::vector<int64_t> &input_dims, const ge::OpDescPtr &op) const;

  // caffe data layer type could be type of `Input` or `DummyData`
  Status ParseParamsForInput(const domi::caffe::LayerParameter &layer, ge::OpDescPtr &op) const;
  Status ParseParamsForDummyData(const domi::caffe::LayerParameter &layer, ge::OpDescPtr &op) const;
};
}  // namespace ge

#endif  // PARSER_CAFFE_CAFFE_DATA_PARSER_H_
