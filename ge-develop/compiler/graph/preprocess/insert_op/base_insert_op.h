/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_PREPROCESS_INSERT_OP_BASE_INSERT_OP_H_
#define GE_GRAPH_PREPROCESS_INSERT_OP_BASE_INSERT_OP_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "framework/common/fmk_error_codes.h"
#include "framework/common/types.h"
#include "framework/common/ge_inner_error_codes.h"
#include "graph/compute_graph.h"
#include "proto/insert_op.pb.h"
#include "proto/om.pb.h"

namespace ge {
class BaseInsertOp {
 public:
  virtual ~BaseInsertOp() = default;

  /// @ingroup ge_omg
  /// @brief Configure the default insertop parameter
  virtual Status SetDefaultParams() = 0;

  /// @ingroup ge_omg
  /// @brief Verify the insertop parameter
  virtual Status ValidateParams() = 0;

  /// @ingroup ge_omg
  /// @brief Insert aipp operator into the network graph
  /// @param [in] graph
  /// @param [in] aippConfigPath aipp
  virtual Status InsertAippToGraph(ge::ComputeGraphPtr &graph, std::string &aippConfigPath,
                                   const uint32_t index) = 0;

  /// @ingroup ge_omg
  /// @brief get aipp mode : static or dyanmic
  /// @param [in] aipp node
  virtual domi::AippOpParams::AippMode GetAippMode() = 0;

 protected:
  /// @ingroup ge_omg
  /// @brief Generate the insert_op operator
  virtual Status GenerateOpDesc(ge::OpDescPtr op_desc) = 0;

  /// @ingroup ge_omg
  /// @brief Get the target operator
  /// @param [in] graph graph
  /// @param [in|out] target_input target operator
  /// @param [in|out] target_edges target edge
  virtual Status GetTargetPosition(ge::ComputeGraphPtr graph, ge::NodePtr &target_input,
                                   std::vector<std::pair<ge::OutDataAnchorPtr, ge::InDataAnchorPtr>> &target_edges) = 0;
};
}  // namespace ge

#endif  // GE_GRAPH_PREPROCESS_INSERT_OP_BASE_INSERT_OP_H_
