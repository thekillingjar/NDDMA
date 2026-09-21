/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/kernel_registry_impl.h"
#include <gtest/gtest.h>
namespace gert {
namespace {
ge::graphStatus TestOutputCreator(const ge::FastNode *, KernelContext *) {
 return ge::GRAPH_SUCCESS;
}
KernelStatus TestFunc(KernelContext *) {
 return 0;
}
std::vector<std::string> TestTraceFunc(const gert::KernelContext *) {
 return {""};
}
}
class KernelRegistryImplUT : public testing::Test {};
TEST_F(KernelRegistryImplUT, RegisterAndFind_Ok_AllFuncRegistered) {
 KernelRegistryImpl registry;
 registry.RegisterKernel("Foo", {{TestFunc, TestOutputCreator, TestTraceFunc}, ""});
 ASSERT_NE(registry.FindKernelFuncs("Foo"), nullptr);
 ASSERT_EQ(registry.FindKernelFuncs("Foo")->run_func, &TestFunc);
 ASSERT_EQ(registry.FindKernelFuncs("Foo")->outputs_creator, &TestOutputCreator);
 ASSERT_EQ(registry.FindKernelFuncs("Foo")->trace_printer, &TestTraceFunc);
}
TEST_F(KernelRegistryImplUT, RegisterAndFind_Ok_OnlyRegisterRunFunc) {
 KernelRegistryImpl registry;
 registry.RegisterKernel("Foo", {{TestFunc, nullptr, nullptr}, ""});
 ASSERT_NE(registry.FindKernelFuncs("Foo"), nullptr);
 ASSERT_EQ(registry.FindKernelFuncs("Foo")->run_func, &TestFunc);
 ASSERT_EQ(registry.FindKernelFuncs("Foo")->outputs_creator, nullptr);
 ASSERT_EQ(registry.FindKernelFuncs("Foo")->trace_printer, nullptr);
}
TEST_F(KernelRegistryImplUT, FailedToFindWhenNotRegister) {
 KernelRegistryImpl registry;
 ASSERT_EQ(registry.FindKernelFuncs("Foo"), nullptr);
 ASSERT_EQ(registry.FindKernelInfo("Foo"), nullptr);
}
TEST_F(KernelRegistryImplUT, GetAll_Ok) {
 KernelRegistryImpl registry;
 registry.RegisterKernel("Foo", {{TestFunc, nullptr, nullptr}, "memory"});
 std::unordered_map<std::string, KernelRegistry::KernelInfo> expect_kernel_infos = {
     {"Foo", {{TestFunc, nullptr, nullptr}, "memory"}},
     {"Bar", {{TestFunc, TestOutputCreator, TestTraceFunc}, "memory"}}
 };
 registry.RegisterKernel("Foo", {{TestFunc, nullptr, nullptr}, "memory"});
 registry.RegisterKernel("Bar", {{TestFunc, TestOutputCreator, TestTraceFunc}, "memory"});
 ASSERT_EQ(registry.GetAll().size(), expect_kernel_infos.size());
 for (const auto &key_to_infos : registry.GetAll()) {
   ASSERT_TRUE(expect_kernel_infos.count(key_to_infos.first) > 0);
   ASSERT_EQ(key_to_infos.second.func.run_func, expect_kernel_infos[key_to_infos.first].func.run_func);
   ASSERT_EQ(key_to_infos.second.critical_section, expect_kernel_infos[key_to_infos.first].critical_section);
 }
}
}  // namespace gert
