/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_COMMON_CUST_AICPU_KERNEL_STORE_H_
#define GE_COMMON_CUST_AICPU_KERNEL_STORE_H_

#include "common/tbe_handle_store/kernel_store.h"
#include "graph/compute_graph.h"
#include <unordered_map>

namespace ge {
class CustAICPUKernelStore : public KernelStore {
 public:
  using KernelStore::KernelStore;
  ~CustAICPUKernelStore() override = default;

  void AddCustAICPUKernel(const CustAICPUKernelPtr &kernel);

  void LoadCustAICPUKernelBinToOpDesc(const std::shared_ptr<OpDesc> &op_desc) const;

  graphStatus BuildKernelSoToOpNameMap(const ComputeGraphPtr &graph);
 private:
  // 保存KernelSo属性与算子名字之间的映射，用于从om中查找算子kernel
  std::unordered_map<std::string, std::string> kernel_so_to_op_name_map_;
};
}  // namespace ge

#endif  // GE_COMMON_CUST_AICPU_KERNEL_STORE_H_
