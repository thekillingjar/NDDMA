/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_GRAPH_UTILS_MEM_UTILS_H_
#define INC_GRAPH_UTILS_MEM_UTILS_H_
#include <memory>
#include <type_traits>
#include "ascendc_ir/ascendc_ir_core/ascendc_ir.h"
namespace ge {
template<typename T>
void CheckAscTensorAttr(T &) {
  static_assert(std::is_same<T, AscTensorAttr>::value, "Expected AscTensorAttr type");
}
/// Usage:
/// Create TQueConfig: MemUtils::CreateTQueConfig(position, depth, buffer_num)
/// TQueConfig BindTensors: config.BindTensors(ascend_tensor1, ascend_tensor2, ...)
class TQueConfig {
  friend class MemUtils;
 public:
  template<typename... Args>
  TQueConfig &BindTensors(Args &&...tensors) {
    int dummy[] = { (CheckAscTensorAttr(std::forward<Args>(tensors)), 0)... };
    (void)dummy;

    int dummy1[] = { (tensors.que = queue_attr_, 0)... };
    (void)dummy1;
    int dummy2[] = { (tensors.buf.id = kIdNone, 0)... };
    (void)dummy2;
    int dummy3[] = { (tensors.mem.position = pos_, 0)... };
    (void)dummy3;
    int dummy4[] = { (tensors.mem.alloc_type = AllocType::kAllocTypeQueue, 0)... };
    (void)dummy4;

    return *this;
  }
  TQueConfig() = default;
 private:
  TQueConfig(const int64_t id, const ge::Position pos, const int64_t depth, const int64_t buf_num);
  MemQueAttr queue_attr_{};
  ge::Position pos_{ge::Position::kPositionInvalid};
};

/// Usage:
/// Create TBufConfig: MemUtils::CreateTBufConfig(position)
/// TBufConfig BindTensors: config.BindTensors(ascend_tensor1, ascend_tensor2, ...)
class TBufConfig {
  friend class MemUtils;
 public:
  template<typename... Args>
  TBufConfig &BindTensors(Args &&...tensors) {
    int dummy[] = { (CheckAscTensorAttr(std::forward<Args>(tensors)), 0)... };
    (void)dummy;

    int dummy1[] = { (tensors.buf = buf_attr_, 0)... };
    (void)dummy1;
    int dummy2[] = { (tensors.que.id = kIdNone, 0)... };
    (void)dummy2;
    int dummy3[] = { (tensors.mem.position = pos_, 0)... };
    (void)dummy3;
    int dummy4[] = { (tensors.mem.alloc_type = AllocType::kAllocTypeBuffer, 0)... };
    (void)dummy4;

    return *this;
  }
  TBufConfig() = default;
 private:
  TBufConfig(const int64_t id, const ge::Position pos);
  MemBufAttr buf_attr_;
  ge::Position pos_;
};

// Only applicable to the three-stage(Tque/Tbuf alloc) ascend ir graph construction
class MemUtils {
 public:
  static TQueConfig CreateTQueConfig(const ge::Position pos, const int64_t depth, const int64_t buf_num);
  static TBufConfig CreateTBufConfig(const ge::Position pos);

  template<typename... Args>
  static void MergeScope(Args &&...tensors) {
    // 修改合并作用域的展开方式
    int dummy[] = { (CheckAscTensorAttr(std::forward<Args>(tensors)), 0)... };
    (void)dummy;
    int dummy1[] = { (tensors.opt.merge_scope = scope_id_, 0)... };
    (void)dummy1;
    scope_id_++;
  }

 private:
  static std::atomic<int64_t> gen_container_id_;
  static std::atomic<int64_t> scope_id_;
};
}  // namespace ge
#endif  // INC_GRAPH_UTILS_MEM_UTILS_H_
