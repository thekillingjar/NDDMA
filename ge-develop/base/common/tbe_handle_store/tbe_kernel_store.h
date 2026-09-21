/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_COMMON_TBE_KERNEL_STORE_H_
#define GE_COMMON_TBE_KERNEL_STORE_H_

#include "common/tbe_handle_store/kernel_store.h"

namespace ge {
class TBEKernelStore : public KernelStore {
 public:
  using KernelStore::KernelStore;
  ~TBEKernelStore() override = default;
  void AddTBEKernel(const TBEKernelPtr &kernel);

  void LoadTBEKernelBinToOpDesc(const std::shared_ptr<ge::OpDesc> &op_desc) const;
};
}  // namespace ge

#endif  // GE_COMMON_TBE_KERNEL_STORE_H_
