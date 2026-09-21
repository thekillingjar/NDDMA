/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include "macro_utils/dt_public_scope.h"
#include "common/tbe_handle_store/cust_aicpu_kernel_store.h"
#include "common/tbe_handle_store/kernel_store.h"
#include "graph/op_kernel_bin.h"
#include "macro_utils/dt_public_unscope.h"
#include "graph/utils/attr_utils.h"

namespace ge {
class UtestCustAicpuKernelStore : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(UtestCustAicpuKernelStore, AddCustAICPUKernel_success) {
  CustAICPUKernelStore ai_cpu_kernel_store;
  //CustAICPUKernelPtr cust_aicpu_kernel;
  const char data[] = "1234";
  vector<char> buffer(data, data + strlen(data));
  ge::OpKernelBinPtr cust_aicpu_kernel = std::make_shared<ge::OpKernelBin>(
        "test", std::move(buffer));
  ai_cpu_kernel_store.AddCustAICPUKernel(cust_aicpu_kernel);
  EXPECT_NE(ai_cpu_kernel_store.FindKernel("test"), nullptr);
}

TEST_F(UtestCustAicpuKernelStore, LoadCustAICPUKernelBinToOpDesc) {
  auto opdesc = std::make_shared<OpDesc>("test", "op");
  CustAICPUKernelStore *ai_cpu_kernel_store = new CustAICPUKernelStore();
  const char data[] = "1234";
  vector<char> buffer(data, data + strlen(data));
  ge::OpKernelBinPtr cust_aicpu_kernel = std::make_shared<ge::OpKernelBin>(
        "test", std::move(buffer));
  //opdesc->SetExtAttr("test", cust_aicpu_kernel);
  (void)ge::AttrUtils::SetStr(opdesc, "kernelSo", "libcust_aicpu_kernel.so");
  ai_cpu_kernel_store->AddCustAICPUKernel(cust_aicpu_kernel);
  ai_cpu_kernel_store->LoadCustAICPUKernelBinToOpDesc(opdesc);
  const ge::OpKernelBinPtr BinPtr =opdesc->TryGetExtAttr<ge::OpKernelBinPtr>(ge::OP_EXTATTR_CUSTAICPU_KERNEL, nullptr);
  EXPECT_EQ(BinPtr, cust_aicpu_kernel);
  delete ai_cpu_kernel_store;
  ai_cpu_kernel_store = nullptr;
}

}
