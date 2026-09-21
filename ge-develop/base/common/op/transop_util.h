/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_COMMON_TRANSOP_UTIL_H_
#define GE_GRAPH_COMMON_TRANSOP_UTIL_H_

#include <string>
#include <unordered_map>

#include "graph/node.h"
#include "common/table_driven.h"

namespace ge {
class TransOpUtil {
 public:
  static bool IsTransOp(const NodePtr &node);

  static bool IsTransOp(const std::string &type);

  static int32_t GetTransOpDataIndex(const NodePtr &node);

  static int32_t GetTransOpDataIndex(const std::string &type);

  /*
   * return True if node input_data_type cast to output_data_type can cause precision loss
   */
  static bool IsPrecisionLoss(const NodePtr &cast_node);

  static std::string TransopMapToString();

 private:
  TransOpUtil();

  ~TransOpUtil() = default;

  static TransOpUtil &Instance();

  std::map<std::string, int32_t> transop_index_map_;
  gert::TableDriven2<static_cast<size_t>(ge::DataType::DT_MAX), static_cast<size_t>(ge::DataType::DT_MAX), bool>
      precision_loss_table_ = gert::TableDriven2<static_cast<size_t>(ge::DataType::DT_MAX),
                                                 static_cast<size_t>(ge::DataType::DT_MAX), bool>(false);
};
}  // namespace ge

#endif  // GE_GRAPH_COMMON_TRANSOP_UTIL_H_
