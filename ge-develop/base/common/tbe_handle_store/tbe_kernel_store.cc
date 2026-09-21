/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/tbe_handle_store/tbe_kernel_store.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"

namespace ge {
void TBEKernelStore::AddTBEKernel(const TBEKernelPtr &kernel) {
  AddKernel(kernel);
}

void TBEKernelStore::LoadTBEKernelBinToOpDesc(const std::shared_ptr<ge::OpDesc> &op_desc) const {
  if (op_desc != nullptr) {
    std::vector<std::string> names_prefix;
    (void)AttrUtils::GetListStr(op_desc, ATTR_NAME_KERNEL_NAMES_PREFIX, names_prefix);
    if (!names_prefix.empty()) {
      const auto LoadMixTbeKernel = [&op_desc, this](const std::string &prefix) {
        std::string kernel_name;
        (void)AttrUtils::GetStr(op_desc, prefix + ATTR_NAME_TBE_KERNEL_NAME, kernel_name);
        const auto &kernel = FindKernel(kernel_name);
        if (kernel != nullptr) {
          const std::string ext_kernel_name = prefix + std::string(OP_EXTATTR_NAME_TBE_KERNEL);
          (void)op_desc->SetExtAttr(ext_kernel_name, kernel);
          GELOGI("LoadTBEKernelBinToOpDesc: Set attr %s for tbe kernel: [%s, %zu] successfully",
                 ext_kernel_name.c_str(), kernel->GetName().c_str(), kernel->GetBinDataSize());
        }
      };
      for (const auto &prefix : names_prefix) {
        LoadMixTbeKernel(prefix);
      }
    } else {
      std::string kernel_name;
      const bool status = AttrUtils::GetStr(op_desc, ATTR_NAME_TBE_KERNEL_NAME_FOR_LOAD, "_kernelname", kernel_name);
      kernel_name = status ? kernel_name : op_desc->GetName();
      const auto &kernel_bin = FindKernel(kernel_name);
      if (kernel_bin != nullptr) {
        GE_IF_BOOL_EXEC(!op_desc->SetExtAttr(ge::OP_EXTATTR_NAME_TBE_KERNEL, kernel_bin),
                        GELOGW("LoadKernelTBEBinToOpDesc: SetExtAttr for kernel_bin failed");)
        GELOGI("Load tbe kernel:%s, %zu", kernel_bin->GetName().c_str(), kernel_bin->GetBinDataSize());
      }
    }
    std::string atomic_kernel_name;
    (void)AttrUtils::GetStr(op_desc, ATOMIC_ATTR_TBE_KERNEL_NAME, atomic_kernel_name);
    if (!atomic_kernel_name.empty()) {
      GELOGI("Get atomic kernel name is %s.", atomic_kernel_name.c_str());
      const auto atomic_kernel_bin = FindKernel(atomic_kernel_name);
      if (atomic_kernel_bin != nullptr) {
        GE_IF_BOOL_EXEC(!op_desc->SetExtAttr(EXT_ATTR_ATOMIC_TBE_KERNEL, atomic_kernel_bin),
                        GELOGW("LoadKernelTBEBinToOpDesc: SetExtAttr for atomic kernel_bin failed");)
        std::string atomic_kernel_bin_attr_name = "_atomic" + std::string(OP_EXTATTR_NAME_TBE_KERNEL);
        GE_IF_BOOL_EXEC(!op_desc->SetExtAttr(atomic_kernel_bin_attr_name, atomic_kernel_bin),
                        GELOGW("LoadKernelTBEBinToOpDesc: SetExtAttr for atomic kernel_bin failed");)
        GELOGI("Load atomic tbe kernel:%s, %zu", atomic_kernel_bin->GetName().c_str(), atomic_kernel_bin->GetBinDataSize());
      }
    }
  }
}
}  // namespace ge
