/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_HYBRID_EXECUTOR_NODE_DONE_MANAGER_H_
#define GE_HYBRID_EXECUTOR_NODE_DONE_MANAGER_H_

#include <condition_variable>
#include <unordered_map>
#include "graph/node.h"
#include "graph/types.h"

namespace ge {
namespace hybrid {
class NodeDoneManager {
 public:
  void NodeDone(const NodePtr &node);

  bool Await(const NodePtr &node);

  void Reset(const NodePtr &node);

  void Destroy();

  void Reset();
 private:
  class Cond {
   public:
    bool IsRelease();
    void Release();
    void Cancel();
    bool Await();
    void Reset();
    static void DevMsgCallback(const char_t *const msg, const uint32_t len) {
      ts_msg_ = std::string(msg, static_cast<size_t>(len));
    }

   private:
    std::mutex cond_mu_;
    std::condition_variable cv_;
    bool is_released_ = false;
    bool is_cancelled_ = false;
    std::string subject_ts_msg_;
    static std::string ts_msg_;
  };

  Cond *GetSubject(const NodePtr &node);
  std::mutex mu_;
  std::unordered_map<NodePtr, Cond> subjects_;
  bool destroyed_ = false;
};
}  // namespace hybrid
}  // namespace ge

#endif // GE_HYBRID_EXECUTOR_NODE_DONE_MANAGER_H_
