/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_GRAPH_OP_KERNEL_BIN_H_
#define INC_GRAPH_OP_KERNEL_BIN_H_

#include <string>
#include <utility>
#include <vector>
#include "graph/types.h"
#include "graph/def_types.h"
#include "graph/host_resource/host_resource.h"

namespace ge {
class OpKernelBin : public HostResource {
 public:
  OpKernelBin(const std::string &name, std::vector<char> &&data) : name_(name), data_(std::move(data)) {}

  ~OpKernelBin() override = default;

  const std::string &GetName() const { return name_; }
  const uint8_t *GetBinData() const { return ge::PtrToPtr<const char_t, const uint8_t>(data_.data()); }
  size_t GetBinDataSize() const { return data_.size(); }
  OpKernelBin(const OpKernelBin &) = delete;
  const OpKernelBin &operator=(const OpKernelBin &) = delete;

 private:
  std::string name_;
  std::vector<char> data_;
};

using OpKernelBinPtr = std::shared_ptr<OpKernelBin>;
constexpr char_t OP_EXTATTR_NAME_TBE_KERNEL[] = "tbeKernel";
constexpr char_t OP_EXTATTR_NAME_THREAD_TBE_KERNEL[] = "thread_tbeKernel";
constexpr char_t OP_EXTATTR_CUSTAICPU_KERNEL[] = "cust_aicpu_kernel";
}  // namespace ge

#endif  // INC_GRAPH_OP_KERNEL_BIN_H_
