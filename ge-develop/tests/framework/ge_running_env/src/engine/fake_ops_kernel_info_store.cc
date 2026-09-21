/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ge/ge_api_error_codes.h"
#include "ge_running_env/fake_ops_kernel_info_store.h"

FAKE_NS_BEGIN

FakeOpsKernelInfoStore::FakeOpsKernelInfoStore(const std::string &info_store_name) : InfoStoreHolder(info_store_name) {}

FakeOpsKernelInfoStore::FakeOpsKernelInfoStore() : InfoStoreHolder() {}

Status FakeOpsKernelInfoStore::Finalize() {
  op_info_map_.clear();
  return SUCCESS;
}

Status FakeOpsKernelInfoStore::Initialize(const std::map<std::string, std::string> &options) { return SUCCESS; }

void FakeOpsKernelInfoStore::GetAllOpsKernelInfo(map<string, OpInfo> &infos) const { infos = op_info_map_; }

bool FakeOpsKernelInfoStore::CheckSupported(const OpDescPtr &op_desc, std::string &) const {
  if (op_desc == nullptr) {
    return false;
  }
  return op_info_map_.count(op_desc->GetType()) > 0;
}

bool FakeOpsKernelInfoStore::CheckSupported(const NodePtr &node, std::string &un_supported_reason,
                                            CheckSupportFlag &flag) const {
  if (node->GetName() == "not_support_dynamic_shape") {
    flag = CheckSupportFlag::kNotSupportDynamicShape;
    return false;
  }
  return CheckSupported(node->GetOpDesc(), un_supported_reason);
}

FAKE_NS_END
