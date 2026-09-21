/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_HYBRID_EXECUTOR_HYBRID_DATA_FLOW_H_
#define GE_HYBRID_EXECUTOR_HYBRID_DATA_FLOW_H_

#include <deque>
#include "hybrid/common/tensor_value.h"
#include "graph/ge_tensor.h"

namespace ge {
namespace hybrid {
using DataFlowTensor = std::pair<TensorValue, GeTensorDesc>;
class TaskContext;
class DataFlowResource {
 public:
  explicit DataFlowResource(const int64_t max_size = std::numeric_limits<int64_t>::max());
  virtual ~DataFlowResource() = default;
  bool IsMaxSizeConst() const;
  bool IsClosed() const;
  bool Empty() const;
  void SetMaxSize(const int64_t max_size);
  void SetMaxSizeConst(const bool max_size_const);
  void SetClosed(const bool is_closed);
  Status EmplaceBack(DataFlowTensor &&value);
  void PopBack();
  const DataFlowTensor &Back();
  void Clear();
 private:
  mutable std::mutex data_flow_res_mutex_;
  int64_t max_size_ = std::numeric_limits<int64_t>::max();
  bool max_size_const_ = false;
  bool is_closed_ = false;
  std::deque<DataFlowTensor> data_;
};

class DataFlowKernelBase {
 public:
  DataFlowKernelBase() = default;
  virtual ~DataFlowKernelBase() = default;
  virtual Status Compute(TaskContext &context, const int64_t handle) = 0;
 protected:
  DataFlowKernelBase(const DataFlowKernelBase &other) = default;
  DataFlowKernelBase &operator=(const DataFlowKernelBase &other) = default;
};
using DataFlowResourcePtr = std::shared_ptr<DataFlowResource>;
using DataFlowKernelBasePtr = std::shared_ptr<DataFlowKernelBase>;
}  // namespace hybrid
}  // namespace ge
#endif  // GE_HYBRID_EXECUTOR_HYBRID_DATA_FLOW_H_
