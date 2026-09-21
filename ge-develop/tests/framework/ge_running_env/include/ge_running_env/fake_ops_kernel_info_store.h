/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef H1EBABA85_7056_48F0_B496_E4DB68E5FED3
#define H1EBABA85_7056_48F0_B496_E4DB68E5FED3

#include "fake_ns.h"
#include "common/opskernel/ops_kernel_info_store.h"
#include "ge/ge_api_types.h"
#include "info_store_holder.h"

FAKE_NS_BEGIN

struct FakeOpsKernelInfoStore : OpsKernelInfoStore, InfoStoreHolder {
  FakeOpsKernelInfoStore(const std::string &kernel_lib_name);
  FakeOpsKernelInfoStore();

 private:
  Status Initialize(const std::map<std::string, std::string> &options) override;
  Status Finalize() override;
  bool CheckSupported(const OpDescPtr &op_desc, std::string &reason) const override;
  bool CheckSupported(const NodePtr &node, std::string &reason, CheckSupportFlag &flag) const override;
  void GetAllOpsKernelInfo(std::map<std::string, ge::OpInfo> &infos) const override;
};

FAKE_NS_END

#endif
