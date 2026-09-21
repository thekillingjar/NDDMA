/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PARSER_COMMON_DATA_OP_PARSER_H_
#define PARSER_COMMON_DATA_OP_PARSER_H_

#include <google/protobuf/text_format.h>
#include <vector>
#include "framework/omg/parser/parser_types.h"
#include "omg/omg_inner_types.h"
#include "proto/om.pb.h"
#include "ge/ge_api_error_codes.h"

#include "graph/attr_value.h"
#include "graph/compute_graph.h"
#include "graph/ge_tensor.h"
#include "graph/op_desc.h"
#include "graph/operator.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/tensor_utils.h"

namespace ge {
/**
 * @ingroup domi_omg
 * @brief Provide a public interface for DataOp
 *
 */
class DataOpParser {
 public:
  virtual ~DataOpParser() {}

 protected:
  /**
   * @ingroup domi_omg
   * @brief parser the Shape information of DataOp
   * @param [in] shape 4D shape information (dimensions)
   * @param [out] op Save converted shap information
   * @return SUCCESS Parsing success
   * @return FAILED Parsing failed
   */
  static Status ParseShape(const std::vector<int64_t> &shape, ge::OpDescPtr op);

 private:
  /**
   * @ingroup domi_omg
   * @brief Convert Input's Shape Information
   * @param [in] 4D shape information (dimensions)
   * @param [out] Save converted shap information
   */
  static Status Init5DInputTensor(const std::vector<int64_t> &shape, ge::GeTensorDesc &tensor_desc);

  /**
   * @ingroup domi_omg
   * @brief Convert Shape of Output
   * @param [in] shape 4D shape information (dimensions)
   * @param [out] output Save converted shap information
   * @return SUCCESS Convert success
   * @return FAILED Convert failed
   */
  static Status Init5DOutputTensor(const std::vector<int64_t> &shape, ge::GeTensorDesc &output);

  /**
   * @ingroup domi_omg
   * @brief 4D shape information (dimensions)4D shape information (dimensions)4D shape information (dimensions)
   * @param [in] 4D shape information (dimensions)
   * @param [out] input Save converted shap information
   */
  static Status InitInputTensor(const std::vector<int64_t> &shape, ge::GeTensorDesc &input);

  /**
   * @ingroup domi_omg
   * @brief Convert Shape of Output
   * @param [in] shape 4D shape information (dimensions)
   * @param [out] output Save converted shap information
   * @return SUCCESS Convert success
   * @return FAILED Convert failed
   */
  static Status InitOutputTensor(const std::vector<int64_t> &shape, ge::GeTensorDesc &output);

  /**
   * @ingroup domi_omg
   * @brief Convert Shape of Output
   * @param [in] shape 4D shape information (dimensions)
   * @param [out] output Save converted shap information
   * @return SUCCESS Convert success
   * @return FAILED Convert failed
   */
  static Status InitNDTensor(const std::vector<int64_t> &shape, ge::DataType data_type, ge::GeTensorDesc &tensor_desc);
};
}  // namespace ge

#endif  // PARSER_COMMON_DATA_OP_PARSER_H_
