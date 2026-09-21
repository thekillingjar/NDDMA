/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_FRAMEWORK_OMG_PARSER_WEIGHTS_PARSER_H_
#define INC_FRAMEWORK_OMG_PARSER_WEIGHTS_PARSER_H_

#include "register/register_error_codes.h"
#include "graph/graph.h"
#include "graph/attr_value.h"
#include "graph/compute_graph.h"
#include "graph/ge_tensor.h"
#include "graph/op_desc.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"

namespace domi {
/**
 * @ingroup domi_omg
 * @brief Weight information resolver
 *
 */
class GE_FUNC_VISIBILITY WeightsParser {
 public:
  /**
   * @ingroup domi_omg
   * @brief Constructor
   */
  WeightsParser() {}

  /**
   * @ingroup domi_omg
   * @brief Deconstructor
   */
  virtual ~WeightsParser() {}

  /**
   * @ingroup domi_omg
   * @brief Analyze weight data
   * @param [in] file Path of weight file after training
   * @param [in|out]  graph Graph for saving weight information after analysis
   * @return SUCCESS
   * @return Others failed
   */
  virtual Status Parse(const char *file, ge::Graph &graph) = 0;

  /**
   * @ingroup domi_omg
   * @brief Parse relevant data from memory and save it to graph
   * @param [in] input Model file memory data
   * @param [in|out] graph A graph for saving the model information after analysis
   * @return SUCCESS
   * @return FAILED
   * @author
   */
  virtual Status ParseFromMemory(const char *input, uint32_t lengt, ge::ComputeGraphPtr &graph) = 0;

  virtual bool HasError() {
    return false;
  }

  virtual Status Save(const std::string &file) {
    (void)file;
    return SUCCESS;
  }

  virtual void Clear() {}
};
}  // namespace domi

#endif  // INC_FRAMEWORK_OMG_PARSER_WEIGHTS_PARSER_H_
