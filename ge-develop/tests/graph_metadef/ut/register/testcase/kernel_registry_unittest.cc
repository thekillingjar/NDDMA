/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/kernel_registry.h"
#include "register/kernel_registry_impl.h"
#include <gtest/gtest.h>

namespace test_gert {
namespace {
ge::graphStatus TestOutputsCreator(const ge::FastNode *, gert::KernelContext *) {
 return ge::GRAPH_SUCCESS;
}
std::vector<std::string> TestTraceFunc(const gert::KernelContext *) {
 return {""};
}
KernelStatus TestFunc1(gert::KernelContext *context) {
 return 0;
}
ge::graphStatus TestOutputsCreator2(const ge::FastNode *, gert::KernelContext *) {
 return ge::GRAPH_SUCCESS;
}
std::vector<std::string> TestTraceFunc2(const gert::KernelContext *) {
 return {""};
}
KernelStatus TestFunc2(gert::KernelContext *) {
 return 0;
}

ge::graphStatus ProfilingInfoFillerTest(const gert::KernelContext *, gert::ProfilingInfoWrapper &) { return 0; }

ge::graphStatus DataDumpInfoFillerTest(const gert::KernelContext *, gert::DataDumpInfoWrapper &) { return 0; }

ge::graphStatus ExceptionDumpInfoFillerTest(const gert::KernelContext *, gert::ExceptionDumpInfoWrapper &) { return 0; }

class FakeRegistry : public gert::KernelRegistry {
public:
 const KernelFuncs *FindKernelFuncs(const std::string &kernel_type) const override {
   static KernelFuncs funcs{nullptr, nullptr, nullptr};
   return &funcs;
 }
};
}  // namespace
class KernelRegistryTest : public testing::Test {
protected:
 void SetUp() override {
   Test::SetUp();
   gert::KernelRegistry::ReplaceKernelRegistry(std::make_shared<gert::KernelRegistryImpl>());
 }
 void TearDown() override {
   gert::KernelRegistry::ReplaceKernelRegistry(nullptr);
 }
};

TEST_F(KernelRegistryTest, RegisterKernel_RegisterSuccess_OnlyRegisterRunFunc) {
 REGISTER_KERNEL(KernelRegistryTest1).RunFunc(TestFunc1);
 auto funcs = gert::KernelRegistry::GetInstance().FindKernelFuncs("KernelRegistryTest1");
 ASSERT_NE(funcs, nullptr);
 EXPECT_EQ(funcs->run_func, &TestFunc1);
}

TEST_F(KernelRegistryTest, RegisterKernel_DefaultFuncOk_OnlyRegisterRunFunc) {
 REGISTER_KERNEL(KernelRegistryTest1).RunFunc(TestFunc1);
 auto funcs = gert::KernelRegistry::GetInstance().FindKernelFuncs("KernelRegistryTest1");
 ASSERT_NE(funcs, nullptr);

 // output creator 默认函数是啥都不干，直接返回成功
 ASSERT_NE(funcs->outputs_creator, nullptr);
 EXPECT_EQ(funcs->outputs_creator(nullptr, nullptr), ge::GRAPH_SUCCESS);

 // trace printer默认值是nullptr
 EXPECT_EQ(funcs->trace_printer, nullptr);
}
TEST_F(KernelRegistryTest, RegisterKernel_Success_OutputCreator) {
 REGISTER_KERNEL(KernelRegistryTest2)
     .OutputsCreator(TestOutputsCreator);
 auto funcs = gert::KernelRegistry::GetInstance().FindKernelFuncs("KernelRegistryTest2");
 ASSERT_NE(funcs, nullptr);
 EXPECT_EQ(funcs->outputs_creator, &TestOutputsCreator);
}
TEST_F(KernelRegistryTest, RegisterKernel_Success_TraceFunc) {
 REGISTER_KERNEL(KernelRegistryTest1).TracePrinter(TestTraceFunc);
 auto funcs = gert::KernelRegistry::GetInstance().FindKernelFuncs("KernelRegistryTest1");
 ASSERT_NE(funcs, nullptr);
 EXPECT_EQ(funcs->trace_printer, &TestTraceFunc);
}
TEST_F(KernelRegistryTest, RegisterKernel_Success_Register_Multiple) {
 REGISTER_KERNEL(KernelRegistryTest1)
     .RunFunc(TestFunc1)
     .OutputsCreator(TestOutputsCreator)
     .TracePrinter(TestTraceFunc);

 REGISTER_KERNEL(KernelRegistryTest2)
     .RunFunc(TestFunc2)
     .OutputsCreator(TestOutputsCreator2)
     .TracePrinter(TestTraceFunc2)
     .ProfilingInfoFiller(ProfilingInfoFillerTest)
     .DataDumpInfoFiller(DataDumpInfoFillerTest)
     .ExceptionDumpInfoFiller(ExceptionDumpInfoFillerTest);

 auto funcs = gert::KernelRegistry::GetInstance().FindKernelFuncs("KernelRegistryTest1");
 ASSERT_NE(funcs, nullptr);
 EXPECT_EQ(funcs->run_func, &TestFunc1);
 EXPECT_EQ(funcs->outputs_creator, &TestOutputsCreator);
 EXPECT_EQ(funcs->trace_printer, &TestTraceFunc);
 EXPECT_EQ(funcs->profiling_info_filler, nullptr);
 EXPECT_EQ(funcs->data_dump_info_filler, nullptr);
 EXPECT_EQ(funcs->exception_dump_info_filler, nullptr);

 funcs = gert::KernelRegistry::GetInstance().FindKernelFuncs("KernelRegistryTest2");
 ASSERT_NE(funcs, nullptr);
 EXPECT_EQ(funcs->run_func, &TestFunc2);
 EXPECT_EQ(funcs->outputs_creator, &TestOutputsCreator2);
 EXPECT_EQ(funcs->trace_printer, &TestTraceFunc2);
 EXPECT_EQ(funcs->profiling_info_filler, &ProfilingInfoFillerTest);
 EXPECT_EQ(funcs->data_dump_info_filler, &DataDumpInfoFillerTest);
 EXPECT_EQ(funcs->exception_dump_info_filler, &ExceptionDumpInfoFillerTest);
}
TEST_F(KernelRegistryTest, RegisterKernel_RegisterOk_SelfDefinedRegistry) {
 // SetUp 中已经是SelfDefinedRegistry了
 REGISTER_KERNEL(KernelRegistryTest1)
     .RunFunc(TestFunc1)
     .OutputsCreator(TestOutputsCreator)
     .TracePrinter(TestTraceFunc);
 auto funcs = gert::KernelRegistry::GetInstance().FindKernelFuncs("KernelRegistryTest1");
 ASSERT_NE(funcs, nullptr);
 EXPECT_EQ(funcs->run_func, &TestFunc1);
 EXPECT_EQ(funcs->outputs_creator, &TestOutputsCreator);
 EXPECT_EQ(funcs->trace_printer, &TestTraceFunc);
}
TEST_F(KernelRegistryTest, SelfDefinedRegistry_RecoveryOk) {
 // 还原为原始的registry
 gert::KernelRegistry::ReplaceKernelRegistry(nullptr);

 // 向原始registry注册
 REGISTER_KERNEL(KernelRegistryTest123)
     .RunFunc(TestFunc1)
     .OutputsCreator(TestOutputsCreator)
     .TracePrinter(TestTraceFunc);

 // replace为自己的实现
 gert::KernelRegistry::ReplaceKernelRegistry(std::make_shared<gert::KernelRegistryImpl>());
 REGISTER_KERNEL(KernelRegistryTest123)
     .RunFunc(TestFunc2)
     .OutputsCreator(TestOutputsCreator2)
     .TracePrinter(TestTraceFunc2);

 auto funcs = gert::KernelRegistry::GetInstance().FindKernelFuncs("KernelRegistryTest123");
 ASSERT_NE(funcs, nullptr);
 EXPECT_EQ(funcs->run_func, &TestFunc2);
 EXPECT_EQ(funcs->outputs_creator, &TestOutputsCreator2);
 EXPECT_EQ(funcs->trace_printer, &TestTraceFunc2);

 // 还原为原始的registry
 gert::KernelRegistry::ReplaceKernelRegistry(nullptr);

 // 原始注册的func还原成功
 funcs = gert::KernelRegistry::GetInstance().FindKernelFuncs("KernelRegistryTest123");
 ASSERT_NE(funcs, nullptr);
 EXPECT_EQ(funcs->run_func, &TestFunc1);
 EXPECT_EQ(funcs->outputs_creator, &TestOutputsCreator);
 EXPECT_EQ(funcs->trace_printer, &TestTraceFunc);
}
TEST_F(KernelRegistryTest, RegisterKernel_NoEffect_RegDeprectedFunc) {
 // SetUp 中已经是SelfDefinedRegistry了
 REGISTER_KERNEL(KernelRegistryTest1)
     .OutputsCreator(TestOutputsCreator);
 auto funcs = gert::KernelRegistry::GetInstance().FindKernelFuncs("KernelRegistryTest1");
 ASSERT_NE(funcs, nullptr);
 ASSERT_NE(funcs->outputs_creator, nullptr);
 EXPECT_EQ(funcs->outputs_creator(nullptr, nullptr), ge::GRAPH_SUCCESS);
}
TEST_F(KernelRegistryTest, RegisterKernel_RegisterSuccess_OnlyRegisterCriticalSection) {
  REGISTER_KERNEL(KernelRegistryTest1).ConcurrentCriticalSectionKey("memory");
  auto kernel_info = gert::KernelRegistry::GetInstance().FindKernelInfo("KernelRegistryTest1");
  ASSERT_NE(kernel_info, nullptr);
  std::string critical_section = kernel_info->critical_section;
  EXPECT_EQ(critical_section, "memory");
}
TEST_F(KernelRegistryTest, RegisterKernel_RegisterSuccess_NotRegisterCriticalSection) {
  REGISTER_KERNEL(KernelRegistryTest1).RunFunc(TestFunc1);
  auto kernel_info =gert::KernelRegistry::GetInstance().FindKernelInfo("KernelRegistryTest1");
  ASSERT_NE(kernel_info, nullptr);
  std::string critical_section = kernel_info->critical_section;
  EXPECT_EQ(critical_section, "");
}
TEST_F(KernelRegistryTest, RegisterKernel_NotRegister_NotFindKernelInfos) {
  EXPECT_EQ(gert::KernelRegistry::GetInstance().FindKernelInfo("KernelRegistryTest1"), nullptr);
}
}  // namespace test_gert
