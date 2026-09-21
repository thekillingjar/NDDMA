/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aicore/launch_kernel/rt_kernel_launch_args_ex.h"
#include <gtest/gtest.h>
#include "faker/kernel_run_context_facker.h"
#include "register/kernel_registry.h"

namespace gert {
namespace {
int64_t GetOffset(void *addr1, void *addr2) {
  auto value2 = reinterpret_cast<size_t>(addr2);
  auto value1 = reinterpret_cast<size_t>(addr1);
  if (value2 >= value1) {
    return static_cast<int64_t>(value2 - value1);
  } else {
    return static_cast<int64_t>(value1 - value2) * -1;
  }
}
using ComputeNodeDesc = RtKernelLaunchArgsEx::ComputeNodeDesc;
void CreateDefaultArgsInfo(ArgsInfosDesc::ArgsInfo *args_info, size_t input_num, size_t output_num) {
  auto node_io_num = input_num + output_num;
  for (size_t idx = 0U; idx < node_io_num; ++idx) {
    int32_t start_index = (idx < input_num) ? idx : (idx - input_num);
    auto arg_type = (idx < input_num) ? ArgsInfosDesc::ArgsInfo::ArgsInfoType::INPUT
                                      : ArgsInfosDesc::ArgsInfo::ArgsInfoType::OUTPUT;
    if (idx == 1) {
      args_info[idx].Init(arg_type, ArgsInfosDesc::ArgsInfo::ArgsInfoFormat::DIRECT_ADDR, -1, 0U);
    } else {
      args_info[idx].Init(arg_type, ArgsInfosDesc::ArgsInfo::ArgsInfoFormat::DIRECT_ADDR, start_index, 1U);
    }
  }
}
std::unique_ptr<uint8_t[]> CreateDefaultArgsInfoDesc(size_t input_num, size_t output_num) {
  const size_t args_info_num = input_num + output_num;
  ArgsInfosDesc::ArgsInfo args_info[args_info_num];
  CreateDefaultArgsInfo(args_info, input_num, output_num);
  size_t total_size = 0U;
  const size_t args_info_size = args_info_num * sizeof(ArgsInfosDesc::ArgsInfo);
  auto args_info_desc_holder = ArgsInfosDesc::Create(args_info_size, total_size);
  auto args_info_desc = reinterpret_cast<ArgsInfosDesc *>(args_info_desc_holder.get());
  if (args_info_size > 0) {
    GELOGD("Copy args info to compute node extended desc mem, size:%lld", args_info_size);
    GE_ASSERT_EOK(memcpy_s(args_info_desc->MutableArgsInfoBase(), args_info_desc->GetArgsInfoSize(), args_info,
                           args_info_size) != EOK);
  }
  args_info_desc->Init(input_num, output_num, input_num, output_num);
  return args_info_desc_holder;
}
}  // namespace
using ComputeNodeDesc = RtKernelLaunchArgsEx::ComputeNodeDesc;
class RtKernelLaunchArgsExUT : public testing::Test {};
TEST_F(RtKernelLaunchArgsExUT, CreateOk) {
  size_t compiled_args_size = 64;
  auto node_desc_holder = malloc(compiled_args_size + sizeof(ComputeNodeDesc));
  auto node_desc = static_cast<ComputeNodeDesc *>(node_desc_holder);
  *node_desc = {.input_num = 1,
                .output_num = 1,
                .workspace_cap = 8,
                .max_tiling_data = 128,
                .need_shape_buffer = false,
                .need_overflow = false,
                .compiled_args_size = compiled_args_size};
  auto args_info_desc_holder = CreateDefaultArgsInfoDesc(node_desc->input_num, node_desc->output_num);
  auto args_info_desc = reinterpret_cast<ArgsInfosDesc *>(args_info_desc_holder.get());
  auto holder = RtKernelLaunchArgsEx::Create(*node_desc, *args_info_desc);
  ASSERT_NE(holder, nullptr);
  free(node_desc_holder);
}

TEST_F(RtKernelLaunchArgsExUT, CreateWithDyInOk) {
  size_t compiled_args_size = 64;
  auto node_desc_holder = malloc(compiled_args_size + sizeof(ComputeNodeDesc));
  auto node_desc = static_cast<ComputeNodeDesc *>(node_desc_holder);
  *node_desc = {.input_num = 1,
      .output_num = 1,
      .workspace_cap = 8,
      .max_tiling_data = 128,
      .need_shape_buffer = false,
      .need_overflow = false,
      .compiled_args_size = compiled_args_size};
  auto args_info_desc_holder = CreateDefaultArgsInfoDesc(node_desc->input_num, node_desc->output_num);
  auto args_info_desc = reinterpret_cast<ArgsInfosDesc *>(args_info_desc_holder.get());
  auto arg_info = reinterpret_cast<ArgsInfosDesc::ArgsInfo *>(args_info_desc->MutableArgsInfoBase());
  arg_info->be_folded = true;
  arg_info->arg_format = ArgsInfosDesc::ArgsInfo::ArgsInfoFormat::FOLDED_DESC_ADDR;
  arg_info->folded_first = true;
  arg_info->start_index = 0;
  arg_info->arg_size = 1;
  auto holder = RtKernelLaunchArgsEx::Create(*node_desc, *args_info_desc);
  ASSERT_NE(holder, nullptr);
  free(node_desc_holder);
}

TEST_F(RtKernelLaunchArgsExUT, OffsetCalcCorrect) {
  size_t compiled_args_size = 64;
  auto node_desc_holder = malloc(compiled_args_size + sizeof(ComputeNodeDesc));
  auto node_desc = reinterpret_cast<ComputeNodeDesc *>(node_desc_holder);
  *node_desc = {.input_num = 1,
                .output_num = 1,
                .workspace_cap = 8,
                .max_tiling_data = 128,
                .need_shape_buffer = false,
                .need_overflow = false,
                .compiled_args_size = compiled_args_size};
  auto args_info_desc_holder = CreateDefaultArgsInfoDesc(node_desc->input_num, node_desc->output_num);
  auto args_info_desc = reinterpret_cast<ArgsInfosDesc *>(args_info_desc_holder.get());
  auto holder = RtKernelLaunchArgsEx::Create(*node_desc, *args_info_desc);
  ASSERT_NE(holder, nullptr);
  auto args = reinterpret_cast<RtKernelLaunchArgsEx *>(holder.get());
  EXPECT_EQ(GetOffset(args->GetArgBase(), args->GetArgsPointer<void>(RtKernelLaunchArgsEx::ArgsType::kArgsCompiledArgs)), 0);
  EXPECT_EQ(GetOffset(args->GetArgBase(), args->GetArgsPointer<void>(RtKernelLaunchArgsEx::ArgsType::kArgsInputsAddr)), 64);
  EXPECT_EQ(GetOffset(args->GetArgBase(), args->GetArgsPointer<void>(RtKernelLaunchArgsEx::ArgsType::kArgsOutputsAddr)), 72);
  EXPECT_EQ(GetOffset(args->GetArgBase(), args->GetArgsPointer<void>(RtKernelLaunchArgsEx::ArgsType::kWorkspacesAddr)), 80);
  EXPECT_EQ(GetOffset(args->GetArgBase(), args->GetArgsPointer<void>(RtKernelLaunchArgsEx::ArgsType::kShapeBufferAddr)), 80);
  EXPECT_EQ(GetOffset(args->GetArgBase(), args->GetArgsPointer<void>(RtKernelLaunchArgsEx::ArgsType::kTilingDataAddr)), 144);
  EXPECT_EQ(GetOffset(args->GetArgBase(), args->GetArgsPointer<void>(RtKernelLaunchArgsEx::ArgsType::kOverflowAddr)), 152);
  EXPECT_EQ(GetOffset(args->GetArgBase(), args->GetArgsPointer<void>(RtKernelLaunchArgsEx::ArgsType::kTilingData)), 152);
  EXPECT_EQ(GetOffset(args->GetArgBase(), args->GetArgsPointer<void>(RtKernelLaunchArgsEx::ArgsType::kHostInputData)), 344);
  EXPECT_EQ(GetOffset(args->GetArgBase(), args->GetArgsPointer<void>(RtKernelLaunchArgsEx::ArgsType::kArgsTypeEnd)), 512);

  EXPECT_EQ(args->GetArgsCap(RtKernelLaunchArgsEx::ArgsType::kArgsCompiledArgs), 64);
  EXPECT_EQ(args->GetArgsCap(RtKernelLaunchArgsEx::ArgsType::kArgsInputsAddr), 8);
  EXPECT_EQ(args->GetArgsCap(RtKernelLaunchArgsEx::ArgsType::kArgsOutputsAddr), 8);
  EXPECT_EQ(args->GetArgsCap(RtKernelLaunchArgsEx::ArgsType::kWorkspacesAddr), 64);
  EXPECT_EQ(args->GetArgsCap(RtKernelLaunchArgsEx::ArgsType::kShapeBufferAddr), 0);
  EXPECT_EQ(args->GetArgsCap(RtKernelLaunchArgsEx::ArgsType::kTilingDataAddr), 8);
  EXPECT_EQ(args->GetArgsCap(RtKernelLaunchArgsEx::ArgsType::kOverflowAddr), 0);
  EXPECT_EQ(args->GetArgsCap(RtKernelLaunchArgsEx::ArgsType::kTilingData), 192);
  EXPECT_EQ(args->GetArgsCap(RtKernelLaunchArgsEx::ArgsType::kHostInputData), 160);
  free(node_desc_holder);
}

TEST_F(RtKernelLaunchArgsExUT, BaseStruCorrect) {
  size_t compiled_args_size = 64;
  auto node_desc_holder = malloc(compiled_args_size + sizeof(ComputeNodeDesc));
  auto node_desc = reinterpret_cast<ComputeNodeDesc *>(node_desc_holder);
  *node_desc = {.input_num = 1,
                .output_num = 1,
                .workspace_cap = 8,
                .max_tiling_data = 128,
                .need_shape_buffer = false,
                .need_overflow = false,
                .compiled_args_size = compiled_args_size};
  auto args_info_desc_holder = CreateDefaultArgsInfoDesc(node_desc->input_num, node_desc->output_num);
  auto args_info_desc = reinterpret_cast<ArgsInfosDesc *>(args_info_desc_holder.get());
  auto holder = RtKernelLaunchArgsEx::Create(*node_desc, *args_info_desc);
  ASSERT_NE(holder, nullptr);
  auto args = reinterpret_cast<RtKernelLaunchArgsEx *>(holder.get());
  auto base = args->GetBase();
  EXPECT_EQ(base->args, args->GetArgBase());
  EXPECT_EQ(base->tilingAddrOffset, 64 + 10 * 8);
  EXPECT_EQ(base->tilingDataOffset, 64 + 11 * 8);
  EXPECT_EQ(base->isNoNeedH2DCopy, false);
  free(node_desc_holder);
}

TEST_F(RtKernelLaunchArgsExUT, SetIoAddr) {
  size_t compiled_args_size = 0;
  auto node_desc_holder = malloc(compiled_args_size + sizeof(ComputeNodeDesc));
  auto node_desc = reinterpret_cast<ComputeNodeDesc *>(node_desc_holder);
  *node_desc = {.input_num = 1,
                .output_num = 1,
                .workspace_cap = 8,
                .max_tiling_data = 128,
                .need_shape_buffer = false,
                .need_overflow = false,
                .compiled_args_size = compiled_args_size};
  auto args_info_desc_holder = CreateDefaultArgsInfoDesc(node_desc->input_num, node_desc->output_num);
  auto args_info_desc = reinterpret_cast<ArgsInfosDesc *>(args_info_desc_holder.get());
  auto holder = RtKernelLaunchArgsEx::Create(*node_desc, *args_info_desc);
  ASSERT_NE(holder, nullptr);
  auto args = reinterpret_cast<RtKernelLaunchArgsEx *>(holder.get());
  args->GetBase()->argsSize = args->GetArgsOffset(RtKernelLaunchArgsEx::ArgsType::kHostInputData) + args->GetHostInputDataSize();
  EXPECT_EQ(args->SetIoAddr(0 * sizeof(TensorAddress), (void *)1024), ge::GRAPH_SUCCESS);
  EXPECT_EQ(args->SetIoAddr(1 * sizeof(TensorAddress), (void *)2048), ge::GRAPH_SUCCESS);

  EXPECT_EQ(*args->GetArgsPointer<TensorAddress>(RtKernelLaunchArgsEx::ArgsType::kArgsInputsAddr), (void *)1024);
  EXPECT_EQ(*args->GetArgsPointer<TensorAddress>(RtKernelLaunchArgsEx::ArgsType::kArgsOutputsAddr), (void *)2048);
  free(node_desc_holder);
}

TEST_F(RtKernelLaunchArgsExUT, SetIoAddrMultipleInputs) {
  ComputeNodeDesc node_desc = {.input_num = 3,
                               .output_num = 1,
                               .workspace_cap = 8,
                               .max_tiling_data = 128,
                               .need_shape_buffer = false,
                               .need_overflow = false,
                               .compiled_args_size = 0};
  auto args_info_desc_holder = CreateDefaultArgsInfoDesc(node_desc.input_num, node_desc.output_num);
  auto args_info_desc = reinterpret_cast<ArgsInfosDesc *>(args_info_desc_holder.get());
  auto holder = RtKernelLaunchArgsEx::Create(node_desc, *args_info_desc);
  ASSERT_NE(holder, nullptr);
  auto args = reinterpret_cast<RtKernelLaunchArgsEx *>(holder.get());
  args->GetBase()->argsSize = args->GetArgsOffset(RtKernelLaunchArgsEx::ArgsType::kHostInputData) + args->GetHostInputDataSize();
  EXPECT_EQ(args->SetIoAddr(0 * sizeof(TensorAddress), (void *)1024), ge::GRAPH_SUCCESS);
  EXPECT_EQ(args->SetIoAddr(1 * sizeof(TensorAddress), (void *)2048), ge::GRAPH_SUCCESS);
  EXPECT_EQ(args->SetIoAddr(2 * sizeof(TensorAddress), (void *)4096), ge::GRAPH_SUCCESS);
  EXPECT_EQ(args->SetIoAddr(3 * sizeof(TensorAddress), (void *)8192), ge::GRAPH_SUCCESS);

  EXPECT_EQ(args->GetArgsPointer<TensorAddress>(RtKernelLaunchArgsEx::ArgsType::kArgsInputsAddr)[0], (void *)1024);
  EXPECT_EQ(args->GetArgsPointer<TensorAddress>(RtKernelLaunchArgsEx::ArgsType::kArgsInputsAddr)[1], (void *)2048);
  EXPECT_EQ(args->GetArgsPointer<TensorAddress>(RtKernelLaunchArgsEx::ArgsType::kArgsInputsAddr)[2], (void *)4096);
  EXPECT_EQ(*args->GetArgsPointer<TensorAddress>(RtKernelLaunchArgsEx::ArgsType::kArgsOutputsAddr), (void *)8192);
}

TEST_F(RtKernelLaunchArgsExUT, UpdateBaseWhenTiling) {
  size_t compiled_args_size = 64;
  auto node_desc_holder = malloc(compiled_args_size + sizeof(ComputeNodeDesc));
  auto node_desc = reinterpret_cast<ComputeNodeDesc *>(node_desc_holder);
  *node_desc = {.input_num = 1,
                .output_num = 1,
                .workspace_cap = 8,
                .max_tiling_data = 128,
                .need_shape_buffer = false,
                .need_overflow = false,
                .compiled_args_size = compiled_args_size};
  auto args_info_desc_holder = CreateDefaultArgsInfoDesc(node_desc->input_num, node_desc->output_num);
  auto args_info_desc = reinterpret_cast<ArgsInfosDesc *>(args_info_desc_holder.get());
  auto holder = RtKernelLaunchArgsEx::Create(*node_desc, *args_info_desc);
  ASSERT_NE(holder, nullptr);
  auto args = reinterpret_cast<RtKernelLaunchArgsEx *>(holder.get());
  args->UpdateBaseByTilingSize(100);
  args->GetTilingData().SetDataSize(100);
  args->UpdateBaseArgsSize();
  EXPECT_TRUE(args->GetBase()->hasTiling);
  EXPECT_EQ(args->GetBase()->argsSize, 64 + 11 * 8 + 100);
  free(node_desc_holder);
}

TEST_F(RtKernelLaunchArgsExUT, UpdateBaseWhenNoTiling) {
  size_t compiled_args_size = 64;
  auto node_desc_holder = malloc(compiled_args_size + sizeof(ComputeNodeDesc));
  auto node_desc = reinterpret_cast<ComputeNodeDesc *>(node_desc_holder);
  *node_desc = {.input_num = 1,
                .output_num = 1,
                .workspace_cap = 8,
                .max_tiling_data = 128,
                .need_shape_buffer = false,
                .need_overflow = false,
                .compiled_args_size = compiled_args_size};
  auto args_info_desc_holder = CreateDefaultArgsInfoDesc(node_desc->input_num, node_desc->output_num);
  auto args_info_desc = reinterpret_cast<ArgsInfosDesc *>(args_info_desc_holder.get());
  auto holder = RtKernelLaunchArgsEx::Create(*node_desc, *args_info_desc);
  ASSERT_NE(holder, nullptr);
  auto args = reinterpret_cast<RtKernelLaunchArgsEx *>(holder.get());
  args->UpdateBaseByTilingSize(0);
  args->GetTilingData().SetDataSize(0);
  args->UpdateBaseArgsSize();
  EXPECT_FALSE(args->GetBase()->hasTiling);
  EXPECT_EQ(args->GetBase()->argsSize, 64 + 11 * 8);
  free(node_desc_holder);
}

TEST_F(RtKernelLaunchArgsExUT, OutputsRefOk) {
  size_t total_size;
  size_t compiled_args_size = 64;
  auto node_desc_holder = ComputeNodeDesc::Create(compiled_args_size, total_size);
  auto node_desc = reinterpret_cast<ComputeNodeDesc *>(node_desc_holder.get());
  *node_desc = {.input_num = 1,
      .output_num = 1,
      .workspace_cap = 8,
      .max_tiling_data = 128,
      .need_shape_buffer = false,
      .need_overflow = false,
      .compiled_args_size = compiled_args_size};
  auto ext_node_desc_holder = CreateDefaultArgsInfoDesc(node_desc->input_num, node_desc->output_num);
  auto context_holder = KernelRunContextFaker()
                            .KernelIONum(2, static_cast<size_t>(AllocLaunchArgOutputs::kNum))
                            .Inputs({node_desc_holder.get(), ext_node_desc_holder.get()})
                            .Build();
  auto context = context_holder.GetContext<KernelContext>();

  auto funcs = KernelRegistry::GetInstance().FindKernelFuncs("AllocLaunchArg");
  ASSERT_NE(funcs, nullptr);
  ASSERT_NE(funcs->outputs_creator, nullptr);
  ASSERT_EQ(funcs->outputs_creator(nullptr, context), ge::GRAPH_SUCCESS);

  auto arg_holder_chain = context->GetOutput(static_cast<size_t>(AllocLaunchArgOutputs::kHolder));
  auto arg_chain = context->GetOutput(static_cast<size_t>(AllocLaunchArgOutputs::kRtArg));
  auto arg_arg_chain = context->GetOutput(static_cast<size_t>(AllocLaunchArgOutputs::kRtArgArgsBase));
  auto tiling_data_chain = context->GetOutput(static_cast<size_t>(AllocLaunchArgOutputs::kTilingDataBase));

  ASSERT_NE(arg_holder_chain, nullptr);
  ASSERT_NE(arg_chain, nullptr);
  ASSERT_NE(arg_arg_chain, nullptr);
  ASSERT_NE(tiling_data_chain, nullptr);

  ASSERT_TRUE(arg_holder_chain->HasDeleter());
  ASSERT_FALSE(arg_chain->HasDeleter());
  ASSERT_FALSE(arg_arg_chain->HasDeleter());
  ASSERT_FALSE(tiling_data_chain->HasDeleter());

  auto arg_holder = arg_holder_chain->GetPointer<RtKernelLaunchArgsEx>();
  auto arg = arg_chain->GetPointer<rtArgsEx_t>();
  auto arg_arg = arg_arg_chain->GetValue<void *>();
  auto tiling_data = tiling_data_chain->GetPointer<TilingData>();

  ASSERT_NE(arg_holder, nullptr);
  ASSERT_NE(arg, nullptr);
  ASSERT_NE(arg_arg, nullptr);
  ASSERT_NE(tiling_data, nullptr);
  EXPECT_NE(tiling_data->GetData(), nullptr);
  EXPECT_EQ(arg_holder->GetBase(), arg);
  EXPECT_EQ(arg_holder->GetArgBase(), arg_arg);
  EXPECT_EQ(&arg_holder->GetTilingData(), tiling_data);

  arg_holder_chain->Set(nullptr, nullptr);
}

TEST_F(RtKernelLaunchArgsExUT, test_calc_compile_args_size_overflow) {
  ComputeNodeDesc node_desc = {.input_num = 1,
                               .output_num = 1,
                               .workspace_cap = 8,
                               .max_tiling_data = 128,
                               .need_shape_buffer = false,
                               .need_overflow = false,
                               .compiled_args_size = INT64_MAX};
  auto args_info_desc_holder = CreateDefaultArgsInfoDesc(node_desc.input_num, node_desc.output_num);
  auto args_info_desc = reinterpret_cast<ArgsInfosDesc *>(args_info_desc_holder.get());
  auto holder = RtKernelLaunchArgsEx::Create(node_desc, *args_info_desc);
  ASSERT_EQ(holder, nullptr);
}

TEST_F(RtKernelLaunchArgsExUT, test_calc_input_overflow) {
  size_t compiled_args_size = 64;
  auto node_desc_holder = malloc(compiled_args_size + sizeof(ComputeNodeDesc));
  auto node_desc = reinterpret_cast<ComputeNodeDesc *>(node_desc_holder);
  *node_desc = {.input_num = INT64_MAX,
                .output_num = 1,
                .workspace_cap = 8,
                .max_tiling_data = 128,
                .need_shape_buffer = false,
                .need_overflow = false,
                .compiled_args_size = compiled_args_size};
  auto args_info_desc_holder = CreateDefaultArgsInfoDesc(0, 0);
  auto args_info_desc = reinterpret_cast<ArgsInfosDesc *>(args_info_desc_holder.get());
  auto holder = RtKernelLaunchArgsEx::Create(*node_desc, *args_info_desc);
  ASSERT_EQ(holder, nullptr);
  free(node_desc_holder);
}

TEST_F(RtKernelLaunchArgsExUT, test_calc_output_overflow) {
  size_t compiled_args_size = 64;
  auto node_desc_holder = malloc(compiled_args_size + sizeof(ComputeNodeDesc));
  auto node_desc = reinterpret_cast<ComputeNodeDesc *>(node_desc_holder);
  *node_desc = {.input_num = 1,
                .output_num = INT64_MAX,
                .workspace_cap = 8,
                .max_tiling_data = 128,
                .need_shape_buffer = false,
                .need_overflow = false,
                .compiled_args_size = compiled_args_size};
  auto args_info_desc_holder = CreateDefaultArgsInfoDesc(0, 0);
  auto args_info_desc = reinterpret_cast<ArgsInfosDesc *>(args_info_desc_holder.get());
  auto holder = RtKernelLaunchArgsEx::Create(*node_desc, *args_info_desc);
  ASSERT_EQ(holder, nullptr);
  free(node_desc_holder);
}

TEST_F(RtKernelLaunchArgsExUT, test_calc_workepace_overflow) {
  size_t compiled_args_size = 64;
  auto node_desc_holder = malloc(compiled_args_size + sizeof(ComputeNodeDesc));
  auto node_desc = reinterpret_cast<ComputeNodeDesc *>(node_desc_holder);
  *node_desc = {.input_num = 1,
                .output_num = 1,
                .workspace_cap = INT64_MAX,
                .max_tiling_data = 128,
                .need_shape_buffer = false,
                .need_overflow = false,
                .compiled_args_size = compiled_args_size};
  auto ext_node_desc_holder = CreateDefaultArgsInfoDesc(node_desc->input_num, node_desc->output_num);
  auto ext_node_desc = reinterpret_cast<ArgsInfosDesc *>(ext_node_desc_holder.get());
  auto holder = RtKernelLaunchArgsEx::Create(*node_desc, *ext_node_desc);
  ASSERT_EQ(holder, nullptr);
  free(node_desc_holder);
}

TEST_F(RtKernelLaunchArgsExUT, test_calc_tiling_overflow) {
  size_t compiled_args_size = 64;
  auto node_desc_holder = malloc(compiled_args_size + sizeof(ComputeNodeDesc));
  auto node_desc = reinterpret_cast<ComputeNodeDesc *>(node_desc_holder);
  *node_desc = {.input_num = 1,
                .output_num = 1,
                .workspace_cap = 2,
                .max_tiling_data = INT64_MAX,
                .need_shape_buffer = false,
                .need_overflow = false,
                .compiled_args_size = compiled_args_size};
  auto args_info_desc_holder = CreateDefaultArgsInfoDesc(node_desc->input_num, node_desc->output_num);
  auto args_info_desc = reinterpret_cast<ArgsInfosDesc *>(args_info_desc_holder.get());
  auto holder = RtKernelLaunchArgsEx::Create(*node_desc, *args_info_desc);
  ASSERT_EQ(holder, nullptr);
  free(node_desc_holder);
}

TEST_F(RtKernelLaunchArgsExUT, test_kernel_run_success) {
  ASSERT_EQ(KernelRegistry::GetInstance().FindKernelFuncs("AllocLaunchArg")->run_func(nullptr), ge::GRAPH_SUCCESS);
}
}  // namespace gert