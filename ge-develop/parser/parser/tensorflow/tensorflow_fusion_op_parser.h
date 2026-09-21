/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OMG_PARSER_TENSORFLOW_TENSORFLOW_FUSION_OP_PARSER_H_
#define OMG_PARSER_TENSORFLOW_TENSORFLOW_FUSION_OP_PARSER_H_

#include <vector>
#include "graph/ge_tensor.h"
#include "omg/parser/op_parser.h"
#include "parser/tensorflow/tensorflow_op_parser.h"
#include "parser/tensorflow/tensorflow_util.h"
#include "proto/tensorflow/graph.pb.h"
#include "proto/tensorflow/node_def.pb.h"

namespace ge {
using google::protobuf::Message;
using domi::tensorflow::NodeDef;
using domi::tensorflow::TensorProto;

/**
 * @ingroup domi_omg
 * @brief Used to parse TensorFlow operator information
 */
class PARSER_FUNC_VISIBILITY TensorFlowFusionOpParser : public TensorFlowOpParser {
 public:
   using TensorFlowOpParser::ParseParams;
 /**
   * @ingroup domi_omg
   * @brief Analytic operator parameters
   * @param [in] v_input_const Operator parameters to be parsed
   * @param [out] op_dest Parsed model data
   * @return SUCCESS Parsing success
   * @return FAILED Parsing failed
   */
  virtual Status ParseParams(const std::vector<const NodeDef *> &v_input_const, ge::NodePtr &op_dest) const;

  /**
   * @ingroup domi_omg
   * @brief Analytic operator parameters
   * @param [in] op_src Parameter data to be parsed
   * @param [out] graph Parsed parameter data
   * @return SUCCESS Parsing success
   * @return FAILED Parsing failed
   */
  Status ParseParams(const Message *op_src, ge::OpDescPtr &op_dest) final;

 protected:
  /**
   * @ingroup domi_omg
   * @brief Parse parameters from const op
   * @param [in] op_src Model data to be parsed
   * @param [out] op_dest Parsed model data
  * @return SUCCESS Parsing success
  * @return FAILED Parsing failed
   *
   */
  // template <class T>
  static Status ParseParamFromConst(const NodeDef *node_def, int32_t &param);

  static Status ParseParamFromConst(const NodeDef *node_def, int32_t &param, int index);

  static Status ParseParamFromConst(const NodeDef *node_def, float &param);

  static Status ParseParamFromConst(const NodeDef *node_def, float &param, int index);

  static Status GetTensorFromNode(const NodeDef *node_def, TensorProto &tensor);

  static Status ParseHalfFromConst(const NodeDef *node_def, float &param, int index = 0);

  static Status ParseWeightFromConst(const NodeDef *node_def, ge::GeTensorPtr &weight);
};
}  // namespace ge

#endif  // OMG_PARSER_TENSORFLOW_TENSORFLOW_FUSION_OP_PARSER_H_
