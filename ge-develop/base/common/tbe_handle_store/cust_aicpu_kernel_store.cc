/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/tbe_handle_store/cust_aicpu_kernel_store.h"
#include "graph_metadef/common/ge_common/util.h"

namespace ge {
void CustAICPUKernelStore::AddCustAICPUKernel(const CustAICPUKernelPtr &kernel) {
  AddKernel(kernel);
}

void CustAICPUKernelStore::LoadCustAICPUKernelBinToOpDesc(const std::shared_ptr<OpDesc> &op_desc) const {
  GE_CHECK_NOTNULL_JUST_RETURN(op_desc);
  std::string kernel_so_name;
  (void)ge::AttrUtils::GetStr(op_desc, "kernelSo", kernel_so_name);
  if (kernel_so_name.empty()) {
    return;
  }
  std::string bin_key = op_desc->GetName();
  auto iter = kernel_so_to_op_name_map_.find(kernel_so_name);
  if (iter != kernel_so_to_op_name_map_.end()) {
    GELOGI("Node: %s has same aicpu kernel with node: %s, kernel_so_name: %s",
        op_desc->GetNamePtr(), iter->second.c_str(), kernel_so_name.c_str());
    bin_key = iter->second;
  }
  const auto &kernel_bin = FindKernel(bin_key);
  if (kernel_bin != nullptr) {
    GE_IF_BOOL_EXEC(!op_desc->SetExtAttr(OP_EXTATTR_CUSTAICPU_KERNEL, kernel_bin),
                    GELOGW("LoadKernelCustAICPUBinToOpDesc: SetExtAttr for kernel_bin failed"));
    GELOGI("Load cust aicpu kernel:%s, bin data size: %zu, op name: %s, bin key: %s",
        kernel_bin->GetName().c_str(), kernel_bin->GetBinDataSize(), op_desc->GetNamePtr(), bin_key.c_str());
  }
}

graphStatus CustAICPUKernelStore::BuildKernelSoToOpNameMap(const ComputeGraphPtr &graph) {
  for (const auto &node : graph->GetAllNodes()) {
    std::string kernel_so_name;
    GE_ASSERT_NOTNULL(node);
    const auto op_desc = node->GetOpDesc();
    GE_ASSERT_NOTNULL(op_desc);
    (void)ge::AttrUtils::GetStr(op_desc, "kernelSo", kernel_so_name);
    if (kernel_so_name.empty()) {
      continue;
    }
    if (FindKernel(op_desc->GetName()) != nullptr) {
      kernel_so_to_op_name_map_.emplace(std::make_pair(kernel_so_name, op_desc->GetName()));
    }
  }
  return SUCCESS;
}
}  // namespace ge
