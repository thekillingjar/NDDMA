/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_TEST_ENGINES_NNENG_GRAPH_CONSTRUCTOR_OPS_KERNEL_INFO_STORE_STUB_H_
#define AIR_TEST_ENGINES_NNENG_GRAPH_CONSTRUCTOR_OPS_KERNEL_INFO_STORE_STUB_H_

#include "common/opskernel/ops_kernel_info_store.h"

namespace fe {
class OpsKernelInfoStoreStub : public ge::OpsKernelInfoStore {
  Status Initialize(const std::map<std::string, std::string> &options) override {
    return fe::SUCCESS;
  }
  Status Finalize() override {
    return fe::SUCCESS;
  }
  void GetAllOpsKernelInfo(std::map<std::string, ge::OpInfo> &infos) const {}
  bool CheckSupported(const ge::OpDescPtr &opDescPtr, std::string &un_supported_reason) const {
    bool supported_flag = true;
    (void)ge::AttrUtils::GetBool(opDescPtr, "supported_flag", supported_flag);
    if (!supported_flag) {
      return false;
    }
    return true;
  }
  bool CheckSupported(const ge::NodePtr &node, std::string &un_supported_reason) const {
    return true;
  }
};
}
#endif  // AIR_TEST_ENGINES_NNENG_GRAPH_CONSTRUCTOR_OPS_KERNEL_INFO_STORE_STUB_H_
