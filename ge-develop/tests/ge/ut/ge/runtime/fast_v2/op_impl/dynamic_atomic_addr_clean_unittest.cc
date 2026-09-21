/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "op_impl/dynamicatomicaddrclean/dynamic_atomic_addr_clean_impl.h"
#include "register/op_impl_registry.h"
#include <gtest/gtest.h>
#include "faker/kernel_run_context_facker.h"
#include "exe_graph/runtime/storage_shape.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/tiling_context.h"
#include "exe_graph/runtime/kernel_context.h"
#include "exe_graph/runtime/tiling_data.h"
#include "exe_graph/runtime/continuous_vector.h"
#include "exe_graph/runtime/tiling_data.h"
namespace gert {
ge::graphStatus TilingForDynamicAtomicAddrClean(TilingContext *context);
}
using namespace gert;
namespace gert_test {
class DynamicAtomicAddrCleanUT : public testing::Test {};
TEST_F(DynamicAtomicAddrCleanUT, TilingOk) {
  size_t clean_size = 4 * 56 * 56 * 16 * 2; // float16

  // compile info
  gert::DynamicAtomicAddrCleanCompileInfo compile_info;
  compile_info.core_num = 2;
  compile_info.ub_size = 126976;
  compile_info.workspace_num = 1;
  compile_info._workspace_index_list = {0};

  auto workspace_sizes_holder = ContinuousVector::Create<size_t>(8);
  auto workspace_sizes = reinterpret_cast<ContinuousVector *>(workspace_sizes_holder.get());
  workspace_sizes->SetSize(1);
  reinterpret_cast<size_t *>(workspace_sizes->MutableData())[0] = 32;

  // tiling data
  auto param = TilingData::CreateCap(2048);
  auto self_workspace_sizes = ContinuousVector::Create<TensorAddress>(8);


  auto holder = gert::KernelRunContextFaker()
                    // 输入信息：一个workspace size，一个需要清空的shape；后面跟着CompileInfo，TilingFunc
                    .KernelIONum(2 + 2, TilingContext::kOutputNum)
                    .IrInputNum(2)
                    .NodeIoNum(2, 0)  // 一个workspace size，一个需要清空的shape；
                    .Inputs({workspace_sizes_holder.get(), (void*)clean_size, &compile_info, nullptr})
                    .Outputs({nullptr, nullptr, nullptr, param.get(), self_workspace_sizes.get(), nullptr})
                    .Build();

  ASSERT_NE(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("DynamicAtomicAddrClean"), nullptr);
  auto tiling_func = gert::TilingForDynamicAtomicAddrClean;
  ASSERT_NE(tiling_func, nullptr);

  EXPECT_EQ(tiling_func(holder.GetContext<gert::TilingContext>()), ge::GRAPH_SUCCESS);
  // todo check tiling result
}

TEST_F(DynamicAtomicAddrCleanUT, TilingParseOk) {
  char *json_str = (char *)"{\"_workspace_index_list\": [0], \"vars\": {\"ub_size\": 126976, \"core_num\": 2, \"workspace_num\": 1}}";
  gert::DynamicAtomicAddrCleanCompileInfo compile_info;
  auto holder = gert::KernelRunContextFaker()
                    .KernelIONum(1, 1)
                    .Inputs({json_str})
                    .Outputs({&compile_info})
                    .Build();

  ASSERT_NE(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("DynamicAtomicAddrClean"), nullptr);
  auto tiling_prepare_func = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("DynamicAtomicAddrClean")->tiling_parse;
  ASSERT_NE(tiling_prepare_func, nullptr);

  EXPECT_EQ(tiling_prepare_func(holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);
  int32_t ub_size = 126976;
  int32_t core_num = 2;
  int32_t workspace_num = 1;
  std::vector<int64_t> _workspace_index_list = {0};
  EXPECT_EQ(compile_info.ub_size, ub_size);
  EXPECT_EQ(compile_info.core_num, core_num);
  EXPECT_EQ(compile_info.workspace_num, workspace_num);
  EXPECT_EQ(compile_info._workspace_index_list[0], _workspace_index_list[0]);
}

TEST_F(DynamicAtomicAddrCleanUT, DetaDepencyOk) {
  ASSERT_NE(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("DynamicAtomicAddrClean"), nullptr);
  EXPECT_FALSE(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("DynamicAtomicAddrClean")->IsInputDataDependency(0));
  EXPECT_FALSE(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("DynamicAtomicAddrClean")->IsInputDataDependency(1));
  EXPECT_FALSE(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("DynamicAtomicAddrClean")->IsInputDataDependency(2));
  EXPECT_FALSE(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("DynamicAtomicAddrClean")->IsInputDataDependency(3));
}
}  // namespace gert_test