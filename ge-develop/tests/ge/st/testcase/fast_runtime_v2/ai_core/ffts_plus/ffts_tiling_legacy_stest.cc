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
#include "kernel/common_kernel_impl/tiling.h"
#include "faker/kernel_run_context_facker.h"
#include "register/kernel_registry.h"
// v1 need
#include "register/op_tiling_registry.h"
#include "graph/utils/op_desc_utils.h"
#include "graph_builder/bg_compatible_utils.h"

namespace gert {
namespace {
struct TestTilingData {
  int64_t a;
  int64_t b;
};

class StubCompileInfoJson : public optiling::CompileInfoBase {
 public:
  StubCompileInfoJson(const std::string &json) : json_str_(json) {}
  ~StubCompileInfoJson() {}
  std::string GetJsonStr() {
    return json_str_;
  };

 private:
  std::string json_str_;
};

bool StubOpTilingFuncV3(const ge::Operator &op, const void *compile_info, optiling::OpRunInfoV2 &op_run_info) {
  op_run_info.SetTilingKey(12);
  return true;
}

bool FailedOpTilingFuncV3(const ge::Operator &op, const void *compile_info, optiling::OpRunInfoV2 &op_run_info) {
  return false;
}

using V3TilingFunc = bool (*)(const ge::Operator &op, const void *compile_info, optiling::OpRunInfoV2 &op_run_info);
using TilingChecker = std::function<void(ge::graphStatus, KernelContext *)>;

optiling::CompileInfoPtr StubOpParseFuncV4(const ge::Operator &op, const ge::AscendString &compileinfo) {
  optiling::CompileInfoPtr info = std::make_shared<StubCompileInfoJson>("testStubOpParseFuncV4");
  return info;
}

bool StubOpTilingFuncV4OnRun(const ge::Operator &op, const optiling::CompileInfoPtr compile_info,
                             optiling::OpRunInfoV2 &op_run_info) {
  // check input output shape
  // input shape  {1, 1, 2, 2, 256}, {1, 2, 2, 256}
  if (op.GetInputDesc(0).GetShape().GetDimNum() != 5) {
    return false;
  }
  if (op.GetInputDesc(0).GetOriginShape().GetDimNum() != 4) {
    return false;
  }
  // output shape {{1, 2, 2, 1, 256}, {2, 2, 1, 256}};
  if (op.GetOutputDesc(0).GetShape().GetDimNum() != 5) {
    return false;
  }
  if (op.GetOutputDesc(0).GetOriginShape().GetDimNum() != 4) {
    return false;
  }
  op_run_info.SetTilingKey(11);
  op_run_info.SetBlockDim(2);
  op_run_info.SetClearAtomic(true);
  op_run_info.SetWorkspaces({1, 2});
  int64_t data1 = 123;
  int64_t data2 = 456;
  op_run_info.AddTilingData(data1);
  op_run_info.AddTilingData(data2);
  return true;
}
}  // namespace
struct TilingCompatibleLegacyKernelSt : public testing::Test {
  void RunTilingAndCheck(V3TilingFunc func, TilingChecker checker) {
    StorageShape input_shape{{1, 2, 2, 256}, {1, 2, 2, 256}};
    StorageShape output_shape{{2, 2, 1, 256}, {2, 2, 1, 256}};
    // build operator with single input and single output
    ge::OpDescPtr desc_ptr = std::make_shared<ge::OpDesc>("test", "Test");
    EXPECT_EQ(desc_ptr->AddInputDesc("x", ge::GeTensorDesc(ge::GeShape({1, -1, 16, 16}), ge::FORMAT_NCHW)),
              ge::GRAPH_SUCCESS);
    EXPECT_EQ(desc_ptr->AddOutputDesc("y", ge::GeTensorDesc(ge::GeShape({1, -1, 8, 8}), ge::FORMAT_NCHW)),
              ge::GRAPH_SUCCESS);
    ge::Operator oprt = ge::OpDescUtils::CreateOperatorFromOpDesc(desc_ptr);
    auto compute_graph = std::make_shared<ge::ComputeGraph>("test");
    auto test_node = compute_graph->AddNode(desc_ptr);
    auto op = ge::OpDescUtils::CreateOperatorFromNode(test_node);

    auto workspace_size_holder = ContinuousVector::Create<size_t>(8);
    auto ws_size = reinterpret_cast<ContinuousVector *>(workspace_size_holder.get());

    // auto run_context = BuildKernelRunContext(6, 5);
    auto param = gert::TilingData::CreateCap(2048);
    std::string op_compile_info_json = "{}";
    uint64_t tiling_version = static_cast<uint64_t>(kernel::TilingVersion::kV3);
    static int x = 1024;
    void *info = &x;
    auto run_context = KernelRunContextFaker()
                           .NodeIoNum(1, 1)
                           .KernelIONum(6, TilingContext::kOutputNum)
                           .Inputs({&op, info, (void *)tiling_version, (void *)func, &input_shape, &output_shape})
                           .IrInputNum(1)
                           .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                           .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                           .Build();
    run_context.GetContext<KernelContext>()->GetOutput(TilingContext::kOutputTilingData)->Set(param.get(), nullptr);
    run_context.GetContext<KernelContext>()->GetOutput(TilingContext::kOutputWorkspace)->Set(ws_size, nullptr);

    auto tiling_func_v1 = KernelRegistry::GetInstance().FindKernelFuncs("CompatibleTilingLegacy");
    auto ret = tiling_func_v1->run_func(run_context);
    checker(ret, run_context.GetContext<KernelContext>());
  }
};

TEST_F(TilingCompatibleLegacyKernelSt, test_tiling_func_v4_success) {
  // mock tiling func
  std::string node_type = "node_with_tling_v4";
  optiling::OpTilingFuncRegistry(node_type, StubOpTilingFuncV4OnRun, StubOpParseFuncV4);

  StorageShape input_shape{{1, 2, 2, 256}, {1, 1, 2, 2, 256}};
  StorageShape output_shape{{2, 2, 1, 256}, {2, 1, 2, 1, 256}};
  // build operator with single input and single output
  ge::OpDescPtr desc_ptr = std::make_shared<ge::OpDesc>("test", "Test");
  EXPECT_EQ(desc_ptr->AddInputDesc("x", ge::GeTensorDesc(ge::GeShape({1, -1, 16, 16}), ge::FORMAT_NCHW)),
            ge::GRAPH_SUCCESS);
  EXPECT_EQ(desc_ptr->AddOutputDesc("y", ge::GeTensorDesc(ge::GeShape({1, -1, 8, 8}), ge::FORMAT_NCHW)),
            ge::GRAPH_SUCCESS);
  desc_ptr->SetOpInferDepends({"x"});
  ge::Operator oprt = ge::OpDescUtils::CreateOperatorFromOpDesc(desc_ptr);
  auto compute_graph = std::make_shared<ge::ComputeGraph>("test");
  auto test_node = compute_graph->AddNode(desc_ptr);
  auto op = ge::OpDescUtils::CreateOperatorFromNode(test_node);

  auto workspace_size_holder = ContinuousVector::Create<size_t>(8);
  auto ws_size = reinterpret_cast<ContinuousVector *>(workspace_size_holder.get());

  // auto run_context = BuildKernelRunContext(6, 5);
  auto param = gert::TilingData::CreateCap(2048);
  std::string op_compile_info_json = "{}";
  uint64_t tiling_version = static_cast<uint64_t>(kernel::TilingVersion::kV4);
  optiling::CompileInfoPtr info = std::make_shared<StubCompileInfoJson>("testStubOpParseFuncV4");
  // build context inputs

  auto run_context =
      gert::KernelRunContextFaker()
          .NodeIoNum(1, 1)
          .KernelIONum(6, TilingContext::kOutputNum)
          .Inputs({&op, &info, (void *)tiling_version, (void *)StubOpTilingFuncV4OnRun, &input_shape, &output_shape})
          .IrInputNum(1)
          .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
          .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
          .Build();
  run_context.GetContext<KernelContext>()->GetOutput(TilingContext::kOutputTilingData)->Set(param.get(), nullptr);
  run_context.GetContext<KernelContext>()->GetOutput(TilingContext::kOutputWorkspace)->Set(ws_size, nullptr);

  auto tiling_func_v1 = KernelRegistry::GetInstance().FindKernelFuncs("CompatibleTilingLegacy");
  ASSERT_EQ(tiling_func_v1->run_func(run_context), ge::GRAPH_SUCCESS);
  auto tiling_key =
      run_context.GetContext<KernelContext>()->GetOutput(TilingContext::kOutputTilingKey)->GetValue<uint64_t>();
  auto block_dims =
      run_context.GetContext<KernelContext>()->GetOutput(TilingContext::kOutputBlockDim)->GetValue<uint64_t>();
  auto clear_atomic_flag =
      run_context.GetContext<KernelContext>()->GetOutput(TilingContext::kOutputAtomicCleanFlag)->GetValue<bool>();
  auto tiling_data_ret =
      run_context.GetContext<KernelContext>()->GetOutput(TilingContext::kOutputTilingData)->GetValue<TilingData *>();
  auto worksapces = run_context.GetContext<KernelContext>()
                        ->GetOutput(TilingContext::kOutputWorkspace)
                        ->GetValue<ContinuousVector *>();
  auto tiling_data = reinterpret_cast<TestTilingData *>(tiling_data_ret->GetData());

  ASSERT_EQ(tiling_key, 11);
  ASSERT_EQ(block_dims, 2);
  ASSERT_EQ(clear_atomic_flag, true);
  ASSERT_EQ(worksapces->GetSize(), 2);
  // workspace size padded to 512
  EXPECT_EQ(reinterpret_cast<const size_t *>(worksapces->GetData())[0], 512);
  EXPECT_EQ(reinterpret_cast<const size_t *>(worksapces->GetData())[1], 512);
  EXPECT_EQ(tiling_data->a, 123);
  EXPECT_EQ(tiling_data->b, 456);
}

TEST_F(TilingCompatibleLegacyKernelSt, Tiling_Success_WhenFuncSuccess) {
  RunTilingAndCheck(StubOpTilingFuncV3, [&](ge::graphStatus result, KernelContext *context) {
    ASSERT_EQ(result, ge::GRAPH_SUCCESS);
    auto tiling_context = reinterpret_cast<TilingContext *>(context);
    ASSERT_EQ(tiling_context->GetTilingKey(), 12);
  });
}
TEST_F(TilingCompatibleLegacyKernelSt, Tiling_Failed_WhenFuncFailed) {
  RunTilingAndCheck(FailedOpTilingFuncV3,
                    [&](ge::graphStatus result, KernelContext *context) { ASSERT_NE(result, ge::GRAPH_SUCCESS); });
}
}  // namespace gert