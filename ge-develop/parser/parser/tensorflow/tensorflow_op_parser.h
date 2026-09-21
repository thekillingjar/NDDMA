/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OMG_PARSER_TENSORFLOW_TENSORFLOW_OP_PARSER_H_
#define OMG_PARSER_TENSORFLOW_TENSORFLOW_OP_PARSER_H_

#if defined(_MSC_VER)
#ifdef FUNC_VISIBILITY
#define PARSER_FUNC_VISIBILITY _declspec(dllexport)
#else
#define PARSER_FUNC_VISIBILITY
#endif
#else
#ifdef FUNC_VISIBILITY
#define PARSER_FUNC_VISIBILITY __attribute__((visibility("default")))
#else
#define PARSER_FUNC_VISIBILITY
#endif
#endif

#include <string>
#include <vector>
#include "framework/omg/parser/op_parser.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/ge_tensor.h"
#include "base/err_msg.h"
#include "graph/node.h"
#include "register/tensor_assign.h"
#include "parser/tensorflow/tensorflow_util.h"
#include "proto/tensorflow/graph.pb.h"
#include "proto/tensorflow/node_def.pb.h"

namespace ge {
/**
 * @ingroup domi_omg
 * @brief used to parse TensorFlow operator information
 */
class PARSER_FUNC_VISIBILITY TensorFlowOpParser : public OpParser {
 public:

  /**
   * @ingroup domi_omg
   * @brief parse params
   * @param [in] op_src        op to be parsed
   * @param [out] op_dest      the parsed op
   * @return SUCCESS           parse success
   * @return FAILED            Parse failed
   *
   */
  Status ParseParams(const Message *op_src, ge::OpDescPtr &op_dest) override {
    (void)op_src;
    (void)op_dest;
    return domi::SUCCESS;
  }

  /**
   * @ingroup domi_omg
   * @brief parse params
   * @param [in] op_src        op to be parsed
   * @param [out] op_dest      the operator
   * @return SUCCESS           parse success
   * @return FAILED            Parse failed
   *
   */
  Status ParseParams(const Message *op_src, ge::Operator &op_dest) override {
    (void)op_src;
    (void)op_dest;
    return domi::SUCCESS;
  }

  /**
   * @ingroup domi_omg
   * @brief parsie weight
   * @param [in] op_src        op to be parsed
   * @param [out] op_dest      the parsed op
   * @return SUCCESS           parsing success
   * @return FAILED            parsing failed
   *
   */
  Status ParseWeights(const Message *op_src, ge::NodePtr &node) final {
    (void)op_src;
    (void)node;
    return domi::SUCCESS;
  }
};
}  // namespace ge

#endif  // OMG_PARSER_TENSORFLOW_TENSORFLOW_OP_PARSER_H_
