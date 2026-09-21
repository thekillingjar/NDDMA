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
#include "stub/kernel_stub.h"

namespace gert {
class KernelStubTest : public testing::Test {};

//UINT32 FakeTiling(KernelContext *context){
//  return 100;
//}
//
//TEST_F(KernelStubTest, test_kern_stub_tiling_faild){
//  KernelStub stub;
//  ASSERT_EQ(stub.SetUp("FakeTiling", FakeTiling), false);
//}
//
//TEST_F(KernelStubTest, test_kern_stub_tiling_success){
//  KernelStub stub;
//  KernelRegistry::KernelFuncs funcs{};
//  ASSERT_EQ(stub.SetUp("Tiling", FakeTiling), true);
//  ASSERT_EQ(KernelRegistryImpl::GetInstance().FindKernelFuncs("Tiling")->run_func(nullptr), 100);
//}

}  // namespace ge
