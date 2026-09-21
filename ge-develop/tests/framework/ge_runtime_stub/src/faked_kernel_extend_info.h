/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_TESTS_FRAMEWORK_GE_RUNTIME_STUB_SRC_FAKED_KERNEL_EXTEND_INFO_H_
#define AIR_CXX_TESTS_FRAMEWORK_GE_RUNTIME_STUB_SRC_FAKED_KERNEL_EXTEND_INFO_H_
#include "exe_graph/runtime/context_extend.h"
#include "stub/gert_runtime_stub.h"
#include <memory>
namespace gert {
class FakedKernelExtendInfo {
 public:
  static void Deleter(FakedKernelExtendInfo *p) {
    delete[] reinterpret_cast<uint8_t *>(p);
  }
  static std::unique_ptr<FakedKernelExtendInfo, void (*)(FakedKernelExtendInfo *)> Build(
      KernelExtendInfo *kernel_extend_info, ge::GeFakeRuntime *gfr) {
    auto size = sizeof(FakedKernelExtendInfo);
    auto p = new uint8_t[size];
    auto faked = reinterpret_cast<FakedKernelExtendInfo *>(p);

    if (kernel_extend_info == nullptr) {
      faked->origin_extend_info_.SetKernelName("Unknown");
      faked->origin_extend_info_.SetKernelType("Unknown");
    } else {
      memcpy(&faked->origin_extend_info_, kernel_extend_info, sizeof(*kernel_extend_info));
    }

    faked->faked_runtime_ = gfr;

    return {faked, FakedKernelExtendInfo::Deleter};
  }
  ge::GeFakeRuntime *GetFakedRuntime() {
    return faked_runtime_;
  }
  const ge::GeFakeRuntime *GetFakedRuntime() const {
    return faked_runtime_;
  }
  KernelExtendInfo &GetExtendInfo() {
    return origin_extend_info_;
  }
  const KernelExtendInfo &GetExtendInfo() const {
    return origin_extend_info_;
  }

 private:
  KernelExtendInfo origin_extend_info_;
  ge::GeFakeRuntime *faked_runtime_;
};
}  // namespace gert

#endif  // AIR_CXX_TESTS_FRAMEWORK_GE_RUNTIME_STUB_SRC_FAKED_KERNEL_EXTEND_INFO_H_
