/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_LOAD_NEW_MODEL_MANAGER_DATA_INPUTER_H_
#define GE_GRAPH_LOAD_NEW_MODEL_MANAGER_DATA_INPUTER_H_

#include <memory>
#include "common/blocking_queue.h"
#include "framework/common/ge_types.h"
#include "framework/common/ge_inner_error_codes.h"
#include "common/model/executor.h"

namespace ge {
/// @ingroup domi_ome
/// @brief manage data input
/// @author
class DataInputer {
 public:
  /// @ingroup domi_ome
  /// @brief constructor
  DataInputer() = default;

  /// @ingroup domi_ome
  /// @brief destructor
  ~DataInputer() = default;

  /// @ingroup domi_ome
  /// @brief add input data
  /// @param [int] input data
  /// @return SUCCESS add successful
  /// @return INTERNAL_ERROR  add failed
  Status Push(const std::shared_ptr<RunArgs> &args) {
    return queue_.Push(args, false) ? SUCCESS : INTERNAL_ERROR;
  }

  /// @ingroup domi_ome
  /// @brief pop input data
  /// @param [out] save popped input data
  /// @return SUCCESS pop success
  /// @return INTERNAL_ERROR  pop fail
  Status Pop(std::shared_ptr<RunArgs> &data) {
    return queue_.Pop(data) ? SUCCESS : INTERNAL_ERROR;
  }

  /// @ingroup domi_ome
  /// @brief stop receiving data, invoke thread at Pop
  void Stop() { queue_.Stop(); }

  uint32_t Size() { return queue_.Size(); }

 private:
  /// @ingroup domi_ome
  /// @brief save input data queue
  BlockingQueue<std::shared_ptr<RunArgs>> queue_;
};
}  // namespace ge
#endif  // GE_GRAPH_LOAD_NEW_MODEL_MANAGER_DATA_INPUTER_H_
