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
#include "register/op_impl_registry.h"
// v1 need
#include "register/op_tiling/op_tiling_constants.h"
#include "register/op_tiling/op_compile_info_manager.h"
#include "register/op_tiling_registry.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/buffer.h"
#include "graph_builder/bg_compatible_utils.h"
#include "common/tiling_fwk_data_helper.h"

namespace gert {
namespace kernel {
ge::graphStatus Tiling(KernelContext *context);
}
namespace {
typedef optiling::CompileInfoPtr (*TilingParseFuncV4)(const ge::Operator &, const ge::AscendString &);
typedef bool (*TilingFuncV4)(const ge::Operator &, const optiling::CompileInfoPtr, optiling::OpRunInfoV2 &);
typedef void *(*TilingParseFuncV3)(const ge::Operator &, const ge::AscendString &);
typedef bool (*TilingFuncV3)(const ge::Operator &, const void *, optiling::OpRunInfoV2 &);

struct TestTilingData {
  int64_t a;
  int64_t b;
};
void *StubOpParseFuncV3(const ge::Operator &op, const ge::AscendString &compileinfo) {
  static void *a = new int(3);
  return a;
}
bool StubOpTilingFuncV3(const ge::Operator &op, const void *compile_info, optiling::OpRunInfoV2 &op_run_info) {
  op_run_info.SetTilingKey(12);
  return true;
}
bool FailedOpTilingFuncV3(const ge::Operator &op, const void *compile_info, optiling::OpRunInfoV2 &op_run_info) {
  return false;
}
using V3TilingFunc = bool (*)(const ge::Operator &op, const void *compile_info, optiling::OpRunInfoV2 &op_run_info);
using TilingChecker = std::function<void(ge::graphStatus, KernelContext *)>;
}  // namespace
struct TilingCompatibleKernelTest : public testing::Test {
  TilingCompatibleKernelTest() {
    tiling = KernelRegistry::GetInstance().FindKernelFuncs("Tiling");
    tilingParse = KernelRegistry::GetInstance().FindKernelFuncs("TilingParse");
    findTilingFunc = KernelRegistry::GetInstance().FindKernelFuncs("FindTilingFunc");
    fake_launch_arg_holder = CreateLaunchArg();
    fake_launch_arg = reinterpret_cast<RtKernelLaunchArgsEx *>(fake_launch_arg_holder.get());
  }
  const KernelRegistry::KernelFuncs *tiling;
  const KernelRegistry::KernelFuncs *tilingParse;
  const KernelRegistry::KernelFuncs *findTilingFunc;
  std::unique_ptr<uint8_t[]> fake_launch_arg_holder;
  RtKernelLaunchArgsEx *fake_launch_arg;

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
    uint64_t tiling_version = static_cast<uint64_t>(kernel::TilingVersion::kV3);
    static int x = 1024;
    void *info = &x;
    kernel::TilingFwkData fwk_data = {.tiling_func = reinterpret_cast<void *>(func), .launch_arg = fake_launch_arg};
    auto run_context = KernelRunContextFaker()
                           .NodeIoNum(1, 1)
                           .KernelIONum(6, static_cast<size_t>(kernel::TilingExOutputIndex::kNum))
                           .Inputs({&op, info, (void *)tiling_version, &fwk_data, &input_shape, &output_shape})
                           .IrInputNum(1)
                           .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                           .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                           .Build();
    run_context.GetContext<KernelContext>()->GetOutput(TilingContext::kOutputWorkspace)->Set(ws_size, nullptr);

    auto tiling_func_v1 = KernelRegistry::GetInstance().FindKernelFuncs("CompatibleTiling");
    auto ret = tiling_func_v1->run_func(run_context);
    checker(ret, run_context.GetContext<KernelContext>());
  }

  void RunFallibleTilingAndCheck(V3TilingFunc func, TilingChecker checker) {
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

    kernel::TilingFwkData fwk_data = {.tiling_func = reinterpret_cast<void *>(func), .launch_arg = fake_launch_arg};
    auto run_context = KernelRunContextFaker()
                           .NodeIoNum(1, 1)
                           .KernelIONum(6, static_cast<size_t>(kernel::TilingExOutputIndex::kNum) + 1)
                           .Inputs({&op, info, (void *)tiling_version, &fwk_data, &input_shape, &output_shape})
                           .IrInputNum(1)
                           .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                           .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                           .Build();
    run_context.GetContext<KernelContext>()->GetOutput(TilingContext::kOutputTilingData)->Set(param.get(), nullptr);
    run_context.GetContext<KernelContext>()->GetOutput(TilingContext::kOutputWorkspace)->Set(ws_size, nullptr);

    auto tiling_func_v1 = KernelRegistry::GetInstance().FindKernelFuncs("FallibleCompatibleTiling");
    auto result = tiling_func_v1->run_func(run_context);
    checker(result, run_context.GetContext<KernelContext>());
  }
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

optiling::CompileInfoPtr StubOpParseFuncV4(const ge::Operator &op, const ge::AscendString &compileinfo) {
  optiling::CompileInfoPtr info = std::make_shared<StubCompileInfoJson>("testStubOpParseFuncV4");
  return info;
}
bool StubOpTilingFuncV4(const ge::Operator &op, const optiling::CompileInfoPtr compile_info,
                        optiling::OpRunInfoV2 &op_run_info) {
  op_run_info.SetTilingKey(11);
  return true;
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

TEST_F(TilingCompatibleKernelTest, test_find_tiling_func_failed) {
  auto run_context = BuildKernelRunContext(1, 3);
  std::string node_type = "node_without_tiling";
  run_context.value_holder[0].Set(const_cast<char *>(node_type.c_str()), nullptr);
  auto find_tiling_func_v1 = KernelRegistry::GetInstance().FindKernelFuncs("FindCompatibleTilingFunc");
  ASSERT_EQ(find_tiling_func_v1->run_func(run_context), ge::GRAPH_FAILED);
}

TEST_F(TilingCompatibleKernelTest, test_find_tiling_func_v4_success) {
  auto run_context = BuildKernelRunContext(1, 3);
  std::string node_type = "node_with_tling_v4";
  run_context.value_holder[0].Set(const_cast<char *>(node_type.c_str()), nullptr);
  // mock tiling func
  optiling::OpTilingFuncRegistry(node_type, StubOpTilingFuncV4, StubOpParseFuncV4);
  auto find_tiling_func_v1 = KernelRegistry::GetInstance().FindKernelFuncs("FindCompatibleTilingFunc");
  ASSERT_EQ(find_tiling_func_v1->run_func(run_context), ge::GRAPH_SUCCESS);
  // check find_tiling_func result, total 3 output
  // (1) tiling version
  // (2) tiling parse func
  // (3) tiling func
  auto tiling_version = run_context.value_holder[1].GetValue<uint64_t>();
  ASSERT_EQ(tiling_version, static_cast<uint64_t>(kernel::TilingVersion::kV4));
  auto tiling_parse_func = reinterpret_cast<TilingParseFuncV4>(run_context.value_holder[2].GetValue<void *>());
  ASSERT_EQ(tiling_parse_func, reinterpret_cast<void *>(&StubOpParseFuncV4));
  // build a operator
  ge::Operator op;
  ge::AscendString compile_info("{}");
  auto compile_info_ptr = tiling_parse_func(op, compile_info);
  auto tiling_func = reinterpret_cast<TilingFuncV4>(run_context.value_holder[3].GetValue<void *>());
  ASSERT_EQ(tiling_func, reinterpret_cast<void *>(&StubOpTilingFuncV4));
  optiling::OpRunInfoV2 op_run_info;
  ASSERT_EQ(tiling_func(op, compile_info_ptr, op_run_info), true);
  ASSERT_EQ(op_run_info.GetTilingKey(), 11);
}

TEST_F(TilingCompatibleKernelTest, test_find_tiling_func_v3_success) {
  auto run_context = BuildKernelRunContext(1, 3);
  std::string node_type = "null_tiling_1";
  run_context.value_holder[0].Set(const_cast<char *>(node_type.c_str()), nullptr);
  // mock tiling func
  optiling::OpTilingFuncRegistry(node_type, StubOpTilingFuncV3, StubOpParseFuncV3);
  auto find_tiling_func_v1 = KernelRegistry::GetInstance().FindKernelFuncs("FindCompatibleTilingFunc");
  ASSERT_EQ(find_tiling_func_v1->run_func(run_context), ge::GRAPH_SUCCESS);
  // check find_tiling_func result, total 3 output
  // (1) tiling version
  // (2) tiling parse func
  // (3) tiling func
  auto tiling_version = run_context.value_holder[1].GetValue<uint64_t>();
  ASSERT_EQ(tiling_version, static_cast<uint64_t>(kernel::TilingVersion::kV3));
  auto tiling_parse_func = reinterpret_cast<TilingParseFuncV3>(run_context.value_holder[2].GetValue<void *>());
  ASSERT_EQ(tiling_parse_func, reinterpret_cast<void *>(&StubOpParseFuncV3));
  // build a operator
  ge::Operator op;
  ge::AscendString compile_info("{}");
  auto compile_info_ptr = tiling_parse_func(op, compile_info);
  auto tiling_func = reinterpret_cast<TilingFuncV3>(run_context.value_holder[3].GetValue<void *>());
  ASSERT_EQ(tiling_func, reinterpret_cast<void *>(&StubOpTilingFuncV3));
  optiling::OpRunInfoV2 op_run_info;
  ASSERT_EQ(tiling_func(op, compile_info_ptr, op_run_info), true);
  ASSERT_EQ(op_run_info.GetTilingKey(), 12);
}
struct OpDescBufferStruct {
  uint8_t *data;
  std::unique_ptr<uint8_t[]> data_holder;
  size_t size;
};
TEST_F(TilingCompatibleKernelTest, test_tiling_parse_func_v4_success) {
  auto run_context = BuildKernelRunContext(6, 1);
  std::string node_type = "null_tiling_2";
  // mock tiling func
  optiling::OpTilingFuncRegistry(node_type, StubOpTilingFuncV4, StubOpParseFuncV4);
  // build operator with single input and single output
  ge::OpDescPtr desc_ptr = std::make_shared<ge::OpDesc>("test", "Test");
  ge::AttrUtils::SetStr(desc_ptr, optiling::COMPILE_INFO_KEY, "{}");
  EXPECT_EQ(desc_ptr->AddInputDesc("x", ge::GeTensorDesc(ge::GeShape({1, 3, 16, 16}), ge::FORMAT_NCHW)),
            ge::GRAPH_SUCCESS);
  EXPECT_EQ(desc_ptr->AddOutputDesc("y", ge::GeTensorDesc(ge::GeShape({1, 3, 8, 8}), ge::FORMAT_NCHW)),
            ge::GRAPH_SUCCESS);
  auto compute_graph = std::make_shared<ge::ComputeGraph>("test");
  auto test_node = compute_graph->AddNode(desc_ptr);
  auto op = ge::OpDescUtils::CreateOperatorFromNode(test_node);

  std::string op_compile_info_json = "{}";
  std::string op_compile_info_key = "{}";
  uint64_t tiling_version = static_cast<uint64_t>(kernel::TilingVersion::kV4);

  // build context inputs
  run_context.value_holder[0].Set(&op, nullptr);
  run_context.value_holder[1].Set(const_cast<char *>(op_compile_info_json.c_str()), nullptr);
  run_context.value_holder[2].Set(const_cast<char *>(op_compile_info_key.c_str()), nullptr);
  run_context.value_holder[3].Set((void *)tiling_version, nullptr);
  run_context.value_holder[4].Set((void *)StubOpParseFuncV4, nullptr);

  auto tiling_parse_func_v1 = KernelRegistry::GetInstance().FindKernelFuncs("CompatibleTilingParse");
  ASSERT_EQ(tiling_parse_func_v1->run_func(run_context), ge::GRAPH_SUCCESS);
  auto compile_info_ptr = reinterpret_cast<optiling::CompileInfoBase *>(run_context.value_holder[6].GetValue<void *>());
  auto stub_compile_info_ptr = (StubCompileInfoJson *)(compile_info_ptr);
  ASSERT_EQ(stub_compile_info_ptr->GetJsonStr(), "testStubOpParseFuncV4");
}

TEST_F(TilingCompatibleKernelTest, test_tiling_parse_func_v3_success) {
  auto run_context = BuildKernelRunContext(6, 1);
  std::string node_type = "null_tiling_3";
  run_context.value_holder[0].Set(const_cast<char *>(node_type.c_str()), nullptr);
  // mock tiling func
  optiling::OpTilingFuncRegistry(node_type, StubOpTilingFuncV3, StubOpParseFuncV3);
  // build operator with single input and single output
  ge::OpDescPtr desc_ptr = std::make_shared<ge::OpDesc>("test", "Test");
  ge::AttrUtils::SetStr(desc_ptr, optiling::COMPILE_INFO_KEY, "{}");
  EXPECT_EQ(desc_ptr->AddInputDesc("x", ge::GeTensorDesc(ge::GeShape({1, 3, 16, 16}), ge::FORMAT_NCHW)),
            ge::GRAPH_SUCCESS);
  EXPECT_EQ(desc_ptr->AddOutputDesc("y", ge::GeTensorDesc(ge::GeShape({1, 3, 8, 8}), ge::FORMAT_NCHW)),
            ge::GRAPH_SUCCESS);
  auto compute_graph = std::make_shared<ge::ComputeGraph>("test");
  auto test_node = compute_graph->AddNode(desc_ptr);
  auto op = ge::OpDescUtils::CreateOperatorFromNode(test_node);

  std::string op_compile_info_json = "{}";
  std::string op_compile_info_key = "{}";
  uint64_t tiling_version = static_cast<uint64_t>(kernel::TilingVersion::kV3);

  // build context inputs
  run_context.value_holder[0].Set(&op, nullptr);
  run_context.value_holder[1].Set(const_cast<char *>(op_compile_info_json.c_str()), nullptr);
  run_context.value_holder[2].Set(const_cast<char *>(op_compile_info_key.c_str()), nullptr);
  run_context.value_holder[3].Set((void *)tiling_version, nullptr);
  run_context.value_holder[4].Set((void *)StubOpParseFuncV3, nullptr);

  auto tiling_parse_func_v1 = KernelRegistry::GetInstance().FindKernelFuncs("CompatibleTilingParse");
  ASSERT_EQ(tiling_parse_func_v1->run_func(run_context), ge::GRAPH_SUCCESS);

  auto oprt = ge::OpDescUtils::CreateOperatorFromNode(test_node);
  void *expect_compile_info = StubOpParseFuncV3(oprt, "test");
  auto compile_info_ptr = reinterpret_cast<void *>(run_context.value_holder[6].GetValue<void *>());
  ASSERT_EQ(compile_info_ptr, expect_compile_info);
}

TEST_F(TilingCompatibleKernelTest, test_tiling_func_v4_success) {
  // mock tiling func
  std::string node_type = "node_with_tling_v4";
  optiling::OpTilingFuncRegistry(node_type, StubOpTilingFuncV4OnRun, StubOpParseFuncV4);

  StorageShape input_shape{{1, 2, 2, 256}, {1,  1, 2, 2, 256}};
  StorageShape output_shape{{2, 2, 1, 256}, {2, 1, 2, 1, 256}};
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
  uint64_t tiling_version = static_cast<uint64_t>(kernel::TilingVersion::kV4);
  optiling::CompileInfoPtr info = std::make_shared<StubCompileInfoJson>("testStubOpParseFuncV4");
  kernel::TilingFwkData fwk_data = {.tiling_func = reinterpret_cast<void *>(StubOpTilingFuncV4OnRun),
                                    .launch_arg = fake_launch_arg};
  // build context inputs
  auto run_context =
      gert::KernelRunContextFaker()
          .NodeIoNum(1, 1)
          .KernelIONum(6, static_cast<size_t>(kernel::TilingExOutputIndex::kNum))
          .Inputs({&op, &info, (void *)tiling_version, &fwk_data, &input_shape, &output_shape})
          .IrInputNum(1)
          .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
          .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
          .Build();
  run_context.GetContext<KernelContext>()->GetOutput(TilingContext::kOutputWorkspace)->Set(ws_size, nullptr);

  auto tiling_func_v1 = KernelRegistry::GetInstance().FindKernelFuncs("CompatibleTiling");
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

TEST_F(TilingCompatibleKernelTest, Tiling_Success_WhenFuncSuccess) {
  RunTilingAndCheck(StubOpTilingFuncV3, [&](ge::graphStatus result, KernelContext *context) {
    ASSERT_EQ(result, ge::GRAPH_SUCCESS);
    auto tiling_context = reinterpret_cast<TilingContext *>(context);
    ASSERT_EQ(tiling_context->GetTilingKey(), 12);
  });
}
TEST_F(TilingCompatibleKernelTest, Tiling_Failed_WhenFuncFailed) {
  RunTilingAndCheck(FailedOpTilingFuncV3,
                    [&](ge::graphStatus result, KernelContext *context) { ASSERT_NE(result, ge::GRAPH_SUCCESS); });
}

TEST_F(TilingCompatibleKernelTest, FallibleTiling_ReturnSuccess_WhenTilingFuncFailed) {
  RunFallibleTilingAndCheck(FailedOpTilingFuncV3, [&](ge::graphStatus result, KernelContext *context) {
    ASSERT_EQ(result, ge::GRAPH_SUCCESS);
    ASSERT_EQ(
        *context->GetOutputPointer<uint32_t>(static_cast<size_t>(kernel::FallibleTilingExOutputIndex::kTilingStatus)),
        1);
  });
}
TEST_F(TilingCompatibleKernelTest, FallibleTiling_ReturnSuccess_WhenTilingFuncSuccess) {
  RunFallibleTilingAndCheck(StubOpTilingFuncV3, [&](ge::graphStatus result, KernelContext *context) {
    ASSERT_EQ(result, ge::GRAPH_SUCCESS);
    ASSERT_EQ(
        *context->GetOutputPointer<uint32_t>(static_cast<size_t>(kernel::FallibleTilingExOutputIndex::kTilingStatus)),
        0);
  });
}
}  // namespace gert